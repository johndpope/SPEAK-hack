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
import time

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


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Use .reshape() instead of .view()
    gradients = gradients.reshape(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_epoch(config, model, dataloader, optimizer_G, optimizer_D, criterion, accelerator, epoch, writer):
    model.train()
    total_loss_G = 0
    total_loss_D = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch+1}")

    # Timing variables
    total_time = 0
    data_loading_time = 0
    forward_pass_time = 0
    d_loss_time = 0
    g_loss_time = 0
    backward_time = 0
    optimizer_step_time = 0
    logging_time = 0

    epoch_start_time = time.time()

    for step, batch in enumerate(dataloader):
        step_start_time = time.time()

        with accelerator.accumulate(model):
            # Data loading time
            data_load_end = time.time()
            data_loading_time += data_load_end - step_start_time

            x_s, x_t = batch["source_image"], batch["target_image"]
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"], batch["emotion_labels_t"]

            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Forward pass time
            forward_start = time.time()
            outputs = model(x_s, x_t)
            x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, _, _ = outputs
            forward_end = time.time()
            forward_pass_time += forward_end - forward_start

            # Discriminator loss time
            d_loss_start = time.time()
            d_real_s = model.D(x_s)
            d_real_t = model.D(x_t)
            d_fake_s = model.D(x_s_recon.detach())
            d_fake_t = model.D(x_t_recon.detach())

            loss_D_real = (
                F.binary_cross_entropy_with_logits(d_real_s, torch.ones_like(d_real_s)) +
                F.binary_cross_entropy_with_logits(d_real_t, torch.ones_like(d_real_t))
            )
            loss_D_fake = (
                F.binary_cross_entropy_with_logits(d_fake_s, torch.zeros_like(d_fake_s)) +
                F.binary_cross_entropy_with_logits(d_fake_t, torch.zeros_like(d_fake_t))
            )

            # Compute gradient penalty
            gradient_penalty_s = compute_gradient_penalty(model.D, x_s, x_s_recon.detach())
            gradient_penalty_t = compute_gradient_penalty(model.D, x_t, x_t_recon.detach())
            gradient_penalty = (gradient_penalty_s + gradient_penalty_t) / 2

            loss_D = loss_D_real + loss_D_fake + config.training.gp_weight * gradient_penalty

            d_loss_end = time.time()
            d_loss_time += d_loss_end - d_loss_start

            # Backward pass time (Discriminator)
            backward_start = time.time()
            accelerator.backward(loss_D)
            backward_end = time.time()
            backward_time += backward_end - backward_start

            # Optimizer step time (Discriminator)
            optim_start = time.time()
            optimizer_D.step()
            optim_end = time.time()
            optimizer_step_time += optim_end - optim_start

            # Train Generator
            optimizer_G.zero_grad()

            # Generator loss time
            g_loss_start = time.time()
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
            g_loss_end = time.time()
            g_loss_time += g_loss_end - g_loss_start

            # Backward pass time (Generator)
            backward_start = time.time()
            accelerator.backward(loss_G)
            backward_end = time.time()
            backward_time += backward_end - backward_start
            
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip_value)
            
            # Optimizer step time (Generator)
            optim_start = time.time()
            optimizer_G.step()
            optim_end = time.time()
            optimizer_step_time += optim_end - optim_start

        if accelerator.sync_gradients:
            progress_bar.update(1)
            total_loss_G += loss_G.detach().float()
            total_loss_D += loss_D.detach().float()

        global_step = epoch * len(dataloader) + step
        if accelerator.is_main_process:
            # Logging time
            log_start = time.time()
            if step % config.training.logging_steps == 0:
                print(f"Epoch {epoch+1}, Step {step}: loss_G = {loss_G.item():.4f}, loss_D = {loss_D.item():.4f}")
                writer.add_scalar('Loss/Train/Generator', loss_G.item(), global_step)
                writer.add_scalar('Loss/Train/Discriminator', loss_D.item(), global_step)
                writer.add_scalar('Loss/Train/Pose_Landmark', l_pose_landmark.item(), global_step)
                writer.add_scalar('Loss/Train/Emotion', l_emotion.item(), global_step)
                writer.add_scalar('Loss/Train/Identity', l_identity.item(), global_step)
                writer.add_scalar('Loss/Train/Reconstruction', l_recon.item(), global_step)
                writer.add_scalar('Loss/Train/GradientPenalty', gradient_penalty.item(), global_step)

            if global_step % config.training.save_image_steps == 0:
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, epoch, step, config.training.output_dir)

            if global_step % config.training.save_steps == 0:
                save_path = os.path.join(config.training.output_dir, f"best_model-epoch-{epoch+1}-{global_step}")
                state_dict = accelerator.unwrap_model(model).get_state_dict()  # Use the custom get_state_dict method
                    
                accelerator.save({
                    'model': state_dict,
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': epoch,
                    'resolution': model.current_resolution,
                    'config': config,
                }, save_path)
            log_end = time.time()
            logging_time += log_end - log_start

        total_time += time.time() - step_start_time

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Print timing statistics
    print(f"\nEpoch {epoch+1} timing statistics:")
    print(f"Total epoch time: {epoch_duration:.2f} seconds")
    print(f"Data loading: {data_loading_time:.2f} seconds ({data_loading_time/epoch_duration*100:.2f}%)")
    print(f"Forward pass: {forward_pass_time:.2f} seconds ({forward_pass_time/epoch_duration*100:.2f}%)")
    print(f"Discriminator loss: {d_loss_time:.2f} seconds ({d_loss_time/epoch_duration*100:.2f}%)")
    print(f"Generator loss: {g_loss_time:.2f} seconds ({g_loss_time/epoch_duration*100:.2f}%)")
    print(f"Backward pass: {backward_time:.2f} seconds ({backward_time/epoch_duration*100:.2f}%)")
    print(f"Optimizer step: {optimizer_step_time:.2f} seconds ({optimizer_step_time/epoch_duration*100:.2f}%)")
    print(f"Logging: {logging_time:.2f} seconds ({logging_time/epoch_duration*100:.2f}%)")
    print(f"Other: {(epoch_duration - total_time):.2f} seconds ({(epoch_duration - total_time)/epoch_duration*100:.2f}%)")

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

    # Load checkpoint if exists
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=accelerator.device)
        model.load_state_dict(checkpoint['model'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_epoch = checkpoint['epoch'] + 1
        current_resolution = checkpoint['resolution']
        config = checkpoint['config']
        model.Gd.set_fourier_state(checkpoint['Gd_fourier_state'])
        print(f"Resumed training from epoch {start_epoch}, resolution {current_resolution}")

    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Start with the highest resolution
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def get_base_dataset(preprocess):
        test =  AffectNetDataset(
            root_dir="/media/oem/12TB/AffectNet/train",
            preprocess=preprocess,
            remove_background=True,
            use_greenscreen=False,
            cache_dir='/media/oem/12TB/AffectNet/train/cache'
        )
        return test
        # return CelebADataset(config.dataset.name, config.dataset.split, preprocess)

      # Load the dataset
    base_dataset = get_base_dataset(preprocess)

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

        preprocess = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            ])

        # preprocess = transforms.Compose([
        #     transforms.Resize((resolution, resolution)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        #     transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomApply([transforms.ElasticTransform(alpha=250.0, sigma=8.0)], p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        # ])
       
        base_dataset = get_base_dataset(preprocess)

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
                    state_dict = accelerator.unwrap_model(model).get_state_dict()  # Use the custom get_state_dict method
                    accelerator.save({
                        'model': state_dict,
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