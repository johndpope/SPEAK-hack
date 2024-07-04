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
from CelebADataset import CelebADataset,ProgressiveDataset,AffectNetDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


def save_debug_images(x_s, x_t, x_s_recon, x_t_recon, step, resolution, output_dir):
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    x_s, x_t = denorm(x_s), denorm(x_t)
    x_s_recon, x_t_recon = denorm(x_s_recon), denorm(x_t_recon)
    
    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"debug_step_{step}_resolution_{resolution}.png"), nrow=4)


# save with emotion labels
def save_emotion_debug_images(x_s, x_t, x_s_recon, x_t_recon, emotion_labels_s, emotion_labels_t, step, resolution, output_dir):
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    def add_emotion_text(img_tensor, emotion_label):
        img = transforms.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((5, 5), f"Emotion: {emotion_label}", (255, 255, 255), font=font)
        return transforms.ToTensor()(img)

    x_s, x_t = denorm(x_s), denorm(x_t)
    x_s_recon, x_t_recon = denorm(x_s_recon), denorm(x_t_recon)
    
    # Add emotion labels to images
    x_s = torch.stack([add_emotion_text(img, label) for img, label in zip(x_s, emotion_labels_s)])
    x_t = torch.stack([add_emotion_text(img, label) for img, label in zip(x_t, emotion_labels_t)])
    x_s_recon = torch.stack([add_emotion_text(img, label) for img, label in zip(x_s_recon, emotion_labels_s)])
    x_t_recon = torch.stack([add_emotion_text(img, label) for img, label in zip(x_t_recon, emotion_labels_t)])

    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"emotion_step_{step}_resolution_{resolution}.png"), nrow=4)


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
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


def log_training_step(writer, loss, l_identity, l_cls, l_pose, l_emotion, l_self, global_step, resolution):
    writer.add_scalar(f'Loss/Total/Resolution_{resolution}', loss.item(), global_step)
    writer.add_scalar(f'Loss/Identity/Resolution_{resolution}', l_identity.item(), global_step)
    writer.add_scalar(f'Loss/Classification/Resolution_{resolution}', l_cls.item(), global_step)
    writer.add_scalar(f'Loss/Pose/Resolution_{resolution}', l_pose.item(), global_step)
    writer.add_scalar(f'Loss/Emotion/Resolution_{resolution}', l_emotion.item(), global_step)
    writer.add_scalar(f'Loss/SelfReconstruction/Resolution_{resolution}', l_self.item(), global_step)

def train_epoch(config, model, dataloader, optimizer_G, optimizer_D, criterion, accelerator, epoch, writer):
    model.train()
    total_loss_G = 0
    total_loss_D = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch+1}")

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            x_s, x_t = batch["source_image"], batch["target_image"]
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]

            # Train Discriminator
            optimizer_D.zero_grad()
            
            outputs = model(x_s, x_t)
            x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, _, _ = outputs

            d_real_s = model.D(x_s)
            d_real_t = model.D(x_t)
            d_fake_s = model.D(x_s_recon.detach())
            d_fake_t = model.D(x_t_recon.detach())

            loss_D = (
                F.binary_cross_entropy_with_logits(d_real_s, torch.ones_like(d_real_s)) +
                F.binary_cross_entropy_with_logits(d_real_t, torch.ones_like(d_real_t)) +
                F.binary_cross_entropy_with_logits(d_fake_s, torch.zeros_like(d_fake_s)) +
                F.binary_cross_entropy_with_logits(d_fake_t, torch.zeros_like(d_fake_t))
            )

            accelerator.backward(loss_D)
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            l_pose_landmark, l_emotion, l_identity, l_recon = criterion(
                x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, 
                emotion_labels_s, emotion_labels_t
            )
            
            d_fake_s = model.D(x_s_recon)
            d_fake_t = model.D(x_t_recon)
            
            loss_G_adv = (
                F.binary_cross_entropy_with_logits(d_fake_s, torch.ones_like(d_fake_s)) +
                F.binary_cross_entropy_with_logits(d_fake_t, torch.ones_like(d_fake_t))
            )

            loss_G = l_pose_landmark + l_emotion + l_identity + l_recon + config.training.stylegan_loss_weight * loss_G_adv

            accelerator.backward(loss_G)
            
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip_value)
            
            optimizer_G.step()
            # Save debug images
            if step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                save_emotion_debug_images(x_s, x_t, x_s_recon, x_t_recon, 
                                  emotion_labels_s, emotion_labels_t, 
                                  step, model.current_resolution, config.training.output_dir)



        if accelerator.sync_gradients:
            progress_bar.update(1)
            total_loss_G += loss_G.detach().float()
            total_loss_D += loss_D.detach().float()

        global_step = epoch * len(dataloader) + step
        if accelerator.is_main_process:
            if step % config.training.logging_steps == 0:
                print(f"Epoch {epoch+1}, Step {step}: loss_G = {loss_G.item():.4f}, loss_D = {loss_D.item():.4f}")
                writer.add_scalar('Loss/Train/Generator', loss_G.item(), global_step)
                writer.add_scalar('Loss/Train/Discriminator', loss_D.item(), global_step)
                writer.add_scalar('Loss/Train/Pose_Landmark', l_pose_landmark.item(), global_step)
                writer.add_scalar('Loss/Train/Emotion', l_emotion.item(), global_step)
                writer.add_scalar('Loss/Train/Identity', l_identity.item(), global_step)
                writer.add_scalar('Loss/Train/Reconstruction', l_recon.item(), global_step)

            if global_step % config.training.save_image_steps == 0:
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, step, config.training.output_dir)


            # In train.py, modify the saving part:
            if (epoch + 1) % config.training.save_epochs == 0 and accelerator.is_main_process:
                save_path = os.path.join(config.training.output_dir, f"best_model-epoch-{epoch+1}")
        
                # In train.py, modify the saving part:
                accelerator.save({
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': epoch,
                    'resolution': model.current_resolution,
                    'config': config,
                },save_path)

        



    return total_loss_G.item() / len(dataloader), total_loss_D.item() / len(dataloader)

def validate(config, model, dataloader, criterion, accelerator, stylegan_loss, current_resolution):
    model.eval()
    total_loss = 0

    
    with torch.no_grad():
        for batch in dataloader:
            x_s, x_t = batch["source_image"], batch["target_image"]
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]

            print(f"Input shapes: x_s: {x_s.shape}, x_t: {x_t.shape}")

            outputs = model(x_s, x_t)
            x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, _, _ = outputs

            print(f"Reconstruction shapes: x_s_recon: {x_s_recon.shape}, x_t_recon: {x_t_recon.shape}")
            print(f"Feature shapes: fi_s: {fi_s.shape}, fe_s: {fe_s.shape}, fp_s: {fp_s.shape}")

            irfd_loss = criterion(
                x_s, x_t, x_s_recon, x_t_recon, 
                fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, 
                emotion_labels_s, emotion_labels_t
            )

            # Flatten and concatenate all features
            fi_s_flat = fi_s.view(x_s.size(0), -1)
            fe_s_flat = fe_s.view(x_s.size(0), -1)
            fp_s_flat = fp_s.view(x_s.size(0), -1)
            
            fi_t_flat = fi_t.view(x_t.size(0), -1)
            fe_t_flat = fe_t.view(x_t.size(0), -1)
            fp_t_flat = fp_t.view(x_t.size(0), -1)

            combined_s = torch.cat([fi_s_flat, fe_s_flat, fp_s_flat], dim=1)
            combined_t = torch.cat([fi_t_flat, fe_t_flat, fp_t_flat], dim=1)

            print(f"Combined feature shapes: combined_s: {combined_s.shape}, combined_t: {combined_t.shape}")

            try:
                fake_s = model.Gd(combined_s, current_resolution)
                fake_t = model.Gd(combined_t, current_resolution)
                print(f"Generated fake shapes: fake_s: {fake_s.shape}, fake_t: {fake_t.shape}")
            except RuntimeError as e:
                print(f"Error in Gd forward pass: {str(e)}")
                print(f"Gd input_dim: {model.Gd.input_dim}")
                raise e

            stylegan_loss_value = stylegan_loss(x_s, fake_s) + stylegan_loss(x_t, fake_t)

            loss = sum(irfd_loss) + config.training.label_balance * stylegan_loss_value
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

    model = IRFD(max_resolution=256)
    
 
    optimizer_G = AdamW(
        [p for n, p in model.named_parameters() if not n.startswith('D.')], 
        lr=config.optimization.learning_rate, 
        betas=(config.optimization.beta1, config.optimization.beta2),
        eps=config.optimization.eps,
        weight_decay=config.optimization.weight_decay
    )
    
    optimizer_D = AdamW(
        model.D.parameters(), 
        lr=config.optimization.learning_rate, 
        betas=(config.optimization.beta1, config.optimization.beta2),
        eps=config.optimization.eps,
        weight_decay=config.optimization.weight_decay
    )

    criterion = IRFDLoss(
        alpha=config.loss.alpha,
        lpips_weight=config.loss.lpips_weight,
        landmark_weight=config.loss.landmark_weight,
        emotion_weight=config.loss.emotion_weight,
        identity_weight=config.loss.identity_weight
    )

    # Check for existing checkpoint
    start_epoch = 0
    latest_checkpoint = None
    if os.path.exists(config.training.output_dir):
        checkpoints = [f for f in os.listdir(config.training.output_dir) if f.startswith("best_model")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(config.training.output_dir, x)))
            latest_checkpoint = os.path.join(config.training.output_dir, latest_checkpoint)
            print(f"👑 Found latest checkpoint: {latest_checkpoint}")


    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Start with the highest resolution
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
        # Load the dataset
    #base_dataset = CelebADataset(config.dataset.name, config.dataset.split, preprocess)

      # Load the dataset
    base_dataset = AffectNetDataset("/media/oem/12TB/AffectNet/train",  preprocess,True)

    train_dataloader = create_progressive_dataloader(config, base_dataset, 64, is_validation=False)
    val_dataloader = create_progressive_dataloader(config, base_dataset, 64, is_validation=True)


    model, optimizer_G, optimizer_D, train_dataloader, val_dataloader, criterion = accelerator.prepare(
        model, optimizer_G, optimizer_D, train_dataloader, val_dataloader, criterion
    )

    stylegan_loss = StyleGANLoss(accelerator.device)
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=config.training.early_stopping_patience)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=config.training.early_stopping_patience)
    writer = SummaryWriter(log_dir=os.path.join(config.training.output_dir, "logs"))

    resolutions = [64, 128, 256]
    epochs_per_resolution = config.training.epochs_per_resolution

    for resolution in resolutions:
        print(f"Training at resolution: {resolution}x{resolution}")
        model.adjust_for_resolution(resolutions[resolutions.index(resolution)])

        train_dataloader = create_progressive_dataloader(config, base_dataset, resolution, is_validation=False)
        val_dataloader = create_progressive_dataloader(config, base_dataset, resolution, is_validation=True)

        model, optimizer_G, optimizer_D, train_dataloader, val_dataloader, criterion = accelerator.prepare(
            model, optimizer_G, optimizer_D, train_dataloader, val_dataloader, criterion
        )
        scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=config.training.early_stopping_patience)
        scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=config.training.early_stopping_patience)

        best_val_loss = float('inf')
        for epoch in range(epochs_per_resolution):
            train_loss_G, train_loss_D = train_epoch(config, model, train_dataloader, optimizer_G, optimizer_D, criterion, accelerator, epoch, writer)
            val_loss = validate(config, model, val_dataloader, criterion, accelerator, stylegan_loss, resolution)

            if accelerator.is_main_process:
                print(f"Resolution: {resolution}, Epoch {epoch+1}/{epochs_per_resolution}: train_loss_G = {train_loss_G:.4f}, train_loss_D = {train_loss_D:.4f}, val_loss = {val_loss:.4f}")
                writer.add_scalar(f'Loss/Train/Generator/Resolution_{resolution}', train_loss_G, epoch)
                writer.add_scalar(f'Loss/Train/Discriminator/Resolution_{resolution}', train_loss_D, epoch)
                writer.add_scalar(f'Loss/Validation/Resolution_{resolution}', val_loss, epoch)

                # Save debug images
                with torch.no_grad():
                    x_s, x_t = next(iter(val_dataloader))["source_image"], next(iter(val_dataloader))["target_image"]
                    outputs = model(x_s, x_t)
                    x_s_recon, x_t_recon = outputs[0], outputs[1]
                    save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, resolution, config.training.output_dir)



            scheduler_G.step(val_loss)
            scheduler_D.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if accelerator.is_main_process:
                    accelerator.save({
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'epoch': epoch,
                        'resolution': resolution,
                        'config': config,
                    }, os.path.join(config.training.output_dir, f"checkpoint-resolution-{resolution}-epoch-{epoch}.pth"))

            

    accelerator.end_training()
    writer.close()
    criterion.__del__()  # Clean up MediaPipe resources

if __name__ == "__main__":
    main()