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
from model import IRFD, IRFDLoss
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from torchvision.utils import save_image
from CelebADataset import CelebADataset, ProgressiveCelebADataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# New imports
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR

def save_debug_images(x_s, x_t, x_s_recon, x_t_recon, step, resolution, output_dir):
    # Denormalize images
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    x_s, x_t = denorm(x_s), denorm(x_t)
    x_s_recon, x_t_recon = denorm(x_s_recon), denorm(x_t_recon)
    
    # Concatenate input and reconstructed images
    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    # Save up to 16 sets of images (64 images total)
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"debug_step_{step}_resolution_{resolution}.png"), nrow=4)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

def get_warmup_scheduler(optimizer, warmup_steps, after_scheduler):
    def lambda_lr(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return after_scheduler.get_last_lr()[0]

    return LambdaLR(optimizer, lambda_lr)
def progressive_train_loop(config, model, base_dataset, optimizer, scheduler, accelerator, writer, criterion, latest_checkpoint=None):
    resolutions = [64, 128, 256, 512]
    epochs_per_resolution = config.training.epochs_per_resolution
    warmup_steps = config.training.warmup_steps
    
    global_step = 0
    start_resolution_index = 0
    start_epoch = 0

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=accelerator.device)
        global_step = checkpoint['global_step']
        last_resolution = checkpoint['resolution']
        start_resolution_index = resolutions.index(last_resolution)
        start_epoch = checkpoint['epoch'] + 1
        if start_epoch >= epochs_per_resolution:
            start_resolution_index += 1
            start_epoch = 0
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming training from resolution {last_resolution}, epoch {start_epoch}")

    # Initialize SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    swa_start = epochs_per_resolution // 2

    # Weight initialization
    model.apply(weight_init)

    # Prepare model, optimizer, and scheduler
    model, optimizer = accelerator.prepare(model, optimizer)

    for resolution_index in range(start_resolution_index, len(resolutions)):
        resolution = resolutions[resolution_index]
        print(f"Training at resolution {resolution}x{resolution}")
        
        dataloader = create_progressive_dataloader(config, base_dataset, resolution)
        
        # Warm-up scheduler
        warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, scheduler)
        
        for epoch in range(start_epoch, epochs_per_resolution):
            model.train()
            total_loss = 0
            progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Resolution {resolution}, Epoch {epoch+1}/{epochs_per_resolution}")

            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    x_s, x_t = batch["source_image"].to(accelerator.device), batch["target_image"].to(accelerator.device)
                    emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"].to(accelerator.device), batch["emotion_labels_t"].to(accelerator.device)
                    # print(f"Input shapes - x_s: {x_s.shape}, x_t: {x_t.shape}")

                    outputs = model(x_s, x_t)
                    loss, l_identity, l_cls, l_pose, l_emotion, l_self = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)
                    
                    # L2 regularization
                    l2_reg = torch.tensor(0., device=accelerator.device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss += config.training.l2_lambda * l2_reg

                    accelerator.backward(loss)
                    
                    # Gradient clipping
                    if config.training.grad_clip:
                        clip_grad_norm_(model.parameters(), config.training.grad_clip_value)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    total_loss += loss.detach().item()

                    if global_step < warmup_steps:
                        warmup_scheduler.step()
                    else:
                        scheduler.step()

                    if writer is not None and accelerator.is_main_process and global_step % config.training.log_steps == 0:
                        log_training_step(writer, loss, l_identity, l_cls, l_pose, l_emotion, l_self, global_step, resolution)

                    if global_step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                        save_debug_images(x_s, x_t, outputs[0], outputs[1], global_step, resolution, config.training.output_dir)

            avg_loss = total_loss / len(dataloader)
            print(f"Resolution {resolution}, Epoch {epoch+1}/{epochs_per_resolution}, Avg Loss: {avg_loss:.4f}")

            # Validation step
            val_loss = validate(model, create_progressive_dataloader(config, base_dataset, resolution), criterion, accelerator)
            print(f"Resolution {resolution}, Epoch {epoch+1}/{epochs_per_resolution}, Validation Loss: {val_loss:.4f}")


            # SWA update
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step(val_loss)

            # Save checkpoint
            if (epoch + 1) % config.training.save_epochs == 0 and accelerator.is_main_process:
                save_path = os.path.join(config.training.output_dir, f"checkpoint-resolution-{resolution}-epoch-{epoch+1}")
                accelerator.save({
                    'epoch': epoch,
                    'resolution': resolution,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'global_step': global_step
                }, save_path)

        # Reset start_epoch for the next resolution
        start_epoch = 0

    # Final SWA update
    torch.optim.swa_utils.update_bn(dataloader, swa_model)
    
    # Save the final SWA model
    if accelerator.is_main_process:
        swa_save_path = os.path.join(config.training.output_dir, "swa_model")
        accelerator.save(swa_model.state_dict(), swa_save_path)

    accelerator.end_training()




def create_progressive_dataloader(config, base_dataset, resolution, is_validation=False):
    progressive_dataset = ProgressiveCelebADataset(base_dataset, resolution)
    
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


def train_epoch(model, dataloader, optimizer, criterion, accelerator, config, writer, global_step):
    model.train()
    total_loss = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            x_s, x_t = batch["source_image"].to(accelerator.device), batch["target_image"].to(accelerator.device)
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"].to(accelerator.device), batch["emotion_labels_t"].to(accelerator.device)

            outputs = model(x_s, x_t)
            loss, l_identity, l_cls, l_pose, l_emotion, l_self = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)

            accelerator.backward(loss)
            
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip_value)
            
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            total_loss += loss.detach().item()

            if writer is not None and accelerator.is_main_process and global_step % config.training.log_steps == 0:
                log_training_step(writer, loss, l_identity, l_cls, l_pose, l_emotion, l_self, global_step,512)

            if global_step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                save_debug_images(x_s, x_t, outputs[0], outputs[1], global_step, config.training.output_dir)

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, accelerator):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x_s, x_t = batch["source_image"].to(accelerator.device), batch["target_image"].to(accelerator.device)
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"].to(accelerator.device), batch["emotion_labels_t"].to(accelerator.device)

            outputs = model(x_s, x_t)
            loss, _, _, _, _, _ = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def main():
    config = OmegaConf.load("config.yaml")
    
    # Create output and log directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    log_dir = os.path.join(config.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Tensorboard logs will be saved to: {log_dir}")

    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Start with the highest resolution
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load the dataset
    base_dataset = CelebADataset(config.dataset.name, config.dataset.split, preprocess)

    # Initialize model, optimizer, and scheduler
    model = IRFD()
    optimizer = Adam(model.parameters(), lr=config.optimization.learning_rate, weight_decay=config.training.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.optimization.lr_patience)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )

    # Initialize tensorboard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(config.training.output_dir, "logs"))

    # Initialize loss function
    criterion = IRFDLoss().to(accelerator.device)

    # Check for existing checkpoint
    latest_checkpoint = None
    if os.path.exists(config.training.output_dir):
        checkpoints = [f for f in os.listdir(config.training.output_dir) if f.startswith("checkpoint-resolution-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(config.training.output_dir, x)))
            latest_checkpoint = os.path.join(config.training.output_dir, latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint}")

    # Run progressive training
    progressive_train_loop(config, model, base_dataset, optimizer, scheduler, accelerator, writer, criterion, latest_checkpoint)

    # Close the SummaryWriter
    writer.close()



if __name__ == "__main__":
    main()


