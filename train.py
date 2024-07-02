import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
from omegaconf import OmegaConf
from datasets import load_dataset
from model import IRFD, IRFDLoss, StyleGANLoss
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from torchvision.utils import save_image
from CelebADataset import CelebADataset,ProgressiveDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import random
import numpy as np


def save_debug_images(x_s, x_t, x_s_recon, x_t_recon, step, resolution, output_dir):
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    x_s, x_t = denorm(x_s), denorm(x_t)
    x_s_recon, x_t_recon = denorm(x_s_recon), denorm(x_t_recon)
    
    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"debug_step_{step}_resolution_{resolution}.png"), nrow=4)

def create_progressive_dataloader(config, base_dataset, resolution, is_validation=False):
    
    progressive_dataset = ProgressiveDataset(base_dataset, resolution)

    # return torch.utils.data.DataLoader(
    #     OverfitDataset('S.png', 'T.png'),
    #     batch_size=1,
    #     num_workers=config.training.num_workers,
    #     pin_memory=True
    # )
    
    # Split the dataset into training and validation
    train_size = int(0.8 * len(progressive_dataset))  # 80% for training
    val_size = len(progressive_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(progressive_dataset, [train_size, val_size])
    
    if is_validation:
        dataset = val_dataset
        batch_size = config.training.eval_batch_size
        shuffle = False
    else:
        dataset = train_dataset
        batch_size = config.training.train_batch_size
        shuffle = True

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True
    )


def log_training_step(writer, loss, l_identity, l_cls, l_pose, l_emotion, l_self, global_step, resolution):
    writer.add_scalar(f'Loss/Total/Resolution_{resolution}', loss.item(), global_step)
    writer.add_scalar(f'Loss/Identity/Resolution_{resolution}', l_identity.item(), global_step)
    writer.add_scalar(f'Loss/Classification/Resolution_{resolution}', l_cls.item(), global_step)
    writer.add_scalar(f'Loss/Pose/Resolution_{resolution}', l_pose.item(), global_step)
    writer.add_scalar(f'Loss/Emotion/Resolution_{resolution}', l_emotion.item(), global_step)
    writer.add_scalar(f'Loss/SelfReconstruction/Resolution_{resolution}', l_self.item(), global_step)

def train_epoch(config, model, dataloader, optimizer, criterion, stylegan_loss, accelerator, epoch, writer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch+1}")

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            x_s, x_t = batch["source_image"], batch["target_image"]
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]

            outputs = model(x_s, x_t)
            x_s_recon, x_t_recon = outputs[0], outputs[1]

            irfd_loss, l_identity, l_cls, l_pose, l_emotion, l_self = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)

            fake_s = model.Gd([model.Ei(x_s).squeeze(-1).squeeze(-1)])
            fake_t = model.Gd([model.Ei(x_t).squeeze(-1).squeeze(-1)])
            stylegan_loss_value = stylegan_loss(x_s, fake_s) + stylegan_loss(x_t, fake_t)

            loss = irfd_loss + config.training.stylegan_loss_weight * stylegan_loss_value

            accelerator.backward(loss)
            
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip_value)
            
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            total_loss += loss.detach().float()

        global_step = epoch * len(dataloader) + step
        if accelerator.is_main_process:
            if step % config.training.logging_steps == 0:
                print(f"Epoch {epoch+1}, Step {step}: loss = {loss.item():.4f}, irfd_loss = {irfd_loss.item():.4f}, stylegan_loss = {stylegan_loss_value.item():.4f}")
                writer.add_scalar('Loss/Train/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Train/IRFD', irfd_loss.item(), global_step)
                writer.add_scalar('Loss/Train/StyleGAN', stylegan_loss_value.item(), global_step)

            if global_step % config.training.save_image_steps == 0:
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, step, config.training.output_dir)

    return total_loss.item() / len(dataloader)


def validate(config, model, dataloader, criterion, stylegan_loss, accelerator):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x_s, x_t = batch["source_image"], batch["target_image"]
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]

            outputs = model(x_s, x_t)
            irfd_loss, _, _, _, _, _ = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)

            fake_s = model.Gd([model.Ei(x_s).squeeze(-1).squeeze(-1)])
            fake_t = model.Gd([model.Ei(x_t).squeeze(-1).squeeze(-1)])
            stylegan_loss_value = stylegan_loss(x_s, fake_s) + stylegan_loss(x_t, fake_t)

            loss = irfd_loss + config.training.label_balance * stylegan_loss_value
            total_loss += loss.detach().float()

    return total_loss.item() / len(dataloader)

def main():
    config = OmegaConf.load("config.yaml")

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with=config.logging.log_with,
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        os.makedirs(config.training.output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(config.training.output_dir, "config.yaml"))

    model = IRFD()
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config.optimization.learning_rate, 
        betas=(config.optimization.beta1, config.optimization.beta2),
        eps=config.optimization.eps,
        weight_decay=config.optimization.weight_decay
    )

    criterion = IRFDLoss()
    stylegan_loss = StyleGANLoss(accelerator.device)

    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Start with the highest resolution
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
        # Load the dataset
    base_dataset = CelebADataset(config.dataset.name, config.dataset.split, preprocess)



    train_dataloader = create_progressive_dataloader(config, base_dataset, 256, is_validation=False)
    val_dataloader = create_progressive_dataloader(config, base_dataset, 256, is_validation=True)


    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.training.early_stopping_patience)
    writer = SummaryWriter(log_dir=os.path.join(config.training.output_dir, "logs"))

    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        train_loss = train_epoch(config, model, train_dataloader, optimizer, criterion, stylegan_loss, accelerator, epoch, writer)
        val_loss = validate(config, model, val_dataloader, criterion, stylegan_loss, accelerator)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{config.training.num_epochs}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

            # Save debug images at the end of each epoch
            with torch.no_grad():
                x_s, x_t = next(iter(val_dataloader))
                outputs = model(x_s, x_t)
                x_s_recon, x_t_recon = outputs[0], outputs[1]
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, 'end', config.training.output_dir)


        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if accelerator.is_main_process:
                accelerator.save({
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                }, os.path.join(config.training.output_dir, "best_model.pth"))


    accelerator.end_training()
    writer.close()

if __name__ == "__main__":
    main()