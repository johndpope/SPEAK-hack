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
from CelebADataset import CelebADataset, ProgressiveDataset,OverfitDataset,AffectNetDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math

# New imports
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR

from torch.optim import AdamW






# Define regular functions for the scale functions
def triangular_scale_fn(x):
    return 1.0

def triangular2_scale_fn(x):
    return 1 / (2.0 ** (x - 1))

def exp_range_scale_fn(x, gamma):
    return gamma ** x


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1., scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        if not isinstance(base_lr, list) and not isinstance(base_lr, tuple):
            base_lr = [base_lr] * len(optimizer.param_groups)
        if not isinstance(max_lr, list) and not isinstance(max_lr, tuple):
            max_lr = [max_lr] * len(optimizer.param_groups)

        self.optimizer = optimizer
        self.base_lrs = base_lr
        self.max_lrs = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentums = [base_momentum] * len(optimizer.param_groups)
        self.max_momentums = [max_momentum] * len(optimizer.param_groups)

        if self.mode == 'triangular':
            self.scale_fn = triangular_scale_fn
        elif self.mode == 'triangular2':
            self.scale_fn = triangular2_scale_fn
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: exp_range_scale_fn(x, self.gamma)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = 1. + self.last_epoch / self.step_size_up - 2 * cycle + 2
        if x <= 1.:
            scale_factor = x
        else:
            scale_factor = (x - 1) / (self.step_size_down / self.step_size_up)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

def get_warmup_scheduler(optimizer, warmup_steps, after_scheduler):
    def lambda_lr(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return after_scheduler.get_last_lr()[0]

    return LambdaLR(optimizer, lambda_lr)

def check_for_nans(tensor, tensor_name=""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        return True
    return False
            
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
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train_step(model, criterion,optimizers, accelerator, x_s, x_t, emotion_labels_s, emotion_labels_t):
    for opt in optimizers.values():
        opt.zero_grad()

    # Forward pass
    with accelerator.accumulate(model):
        outputs = model(x_s, x_t)

        # criterin = IRFDLoss
        l_pose, l_emotion, l_identity, l_recon = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)

        # Accumulate losses
        total_loss = l_pose + l_emotion + l_identity + l_recon

        # Backward pass (single pass for all losses)
        accelerator.backward(total_loss)

        # Step optimizers
        for opt in optimizers.values():
            opt.step()

    return outputs, l_pose.item(), l_emotion.item(), l_identity.item(), l_recon.item()


    #     # Backward pass for pose encoder (Ep)
    #     pose_loss = l_landmark #+ 0.1 * l_recon  # Adjust the 0.1 factor as needed
    #     accelerator.backward(pose_loss)
    #     optimizers['pose'].step()
    #     model.Ep.zero_grad()

    #     # Backward pass for emotion encoder (Ee)
    #     emotion_loss = l_emotion# + 0.1 * l_recon  # Adjust the 0.1 factor as needed
    #     accelerator.backward(emotion_loss)
    #     optimizers['emotion'].step()
    #     model.Ee.zero_grad()

    #     # Backward pass for identity encoder (Ei)
    #     identity_loss = l_identity #+ 0.1 * l_recon  # Adjust the 0.1 factor as needed
    #     accelerator.backward(identity_loss)
    #     optimizers['identity'].step()
    #     model.Ei.zero_grad()

    #     # Backward pass for reconstruction (affects all parts of the model)
    #     accelerator.backward(l_recon)
    #     optimizers['other'].step()
    #     optimizers['pose'].step()
    #     optimizers['emotion'].step()
    #     optimizers['identity'].step()

    # return outputs,l_landmark.item(), l_emotion.item(), l_identity.item(), l_recon.item()

def progressive_irfd_train_loop(config, model, base_dataset, optimizers, accelerator, writer, criterion, latest_checkpoint=None):
    resolutions = [256]
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
        
        # Load model state dict while ignoring unexpected keys
        model_dict = model.state_dict()
        checkpoint_model_dict = checkpoint['model_state_dict']
        filtered_dict = {k: v for k, v in checkpoint_model_dict.items() if k in model_dict}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        # Load optimizers state dicts
        for k, opt in optimizers.items():
            opt.load_state_dict(checkpoint[f'optimizer_{k}_state_dict'])
        
        print(f"Resuming training from resolution {last_resolution}, epoch {start_epoch}")
        print(f"Loaded {len(filtered_dict)} / {len(checkpoint_model_dict)} keys from checkpoint")

    # Weight initialization
    model.apply(weight_init)

    # Prepare model and optimizers
    model, optimizers = accelerator.prepare(model, optimizers)

    for resolution_index in range(start_resolution_index, len(resolutions)):
        resolution = resolutions[resolution_index]
        print(f"Training at resolution {resolution}x{resolution}")
        
        train_dataloader = create_progressive_dataloader(config, base_dataset, resolution, is_validation=False)
        val_dataloader = create_progressive_dataloader(config, base_dataset, resolution, is_validation=True)

        # Initialize cyclical learning rate scheduler
        base_lr = config.optimization.learning_rate
        max_lr = 0.0001  # from the white paper
        step_size = epochs_per_resolution * len(train_dataloader) // 2  # Half the epochs per resolution
        schedulers = {k: CyclicLR(opt, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode='triangular2')
                      for k, opt in optimizers.items()}

        # Warm-up scheduler
        warmup_schedulers = {k: get_warmup_scheduler(opt, warmup_steps, schedulers[k])
                             for k, opt in optimizers.items()}
        
        for epoch in range(start_epoch, epochs_per_resolution):
            model.train()
            total_loss = 0
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Resolution {resolution}, Epoch {epoch+1}/{epochs_per_resolution}")

            for step, batch in enumerate(train_dataloader):
                x_s, x_t = batch["source_image"].to(accelerator.device), batch["target_image"].to(accelerator.device)
                emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"].to(accelerator.device), batch["emotion_labels_t"].to(accelerator.device)
                
                try:
                    outputs,l_landmark, l_emotion, l_identity, l_recon = train_step(model,criterion, optimizers, accelerator, x_s, x_t, emotion_labels_s, emotion_labels_t)
                    loss = l_landmark + l_emotion + l_identity + l_recon
                    
                    # if torch.isnan(loss) or torch.isinf(loss):
                    #     print(f"Loss is {loss}, skipping this batch")
                    #     continue

                except Exception as e:
                    print("error:", e)
                    save_image(x_s, os.path.join(config.training.output_dir, "x_s_failed.png"))
                    save_image(x_t, os.path.join(config.training.output_dir, "x_t_failed.png"))
                    raise ValueError("Error in x_s or x_t - check x_s_failed.png and x_t_failed.png") 

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    total_loss += loss

                    if global_step < warmup_steps:
                        for ws in warmup_schedulers.values():
                            ws.step()
                    else:
                        for s in schedulers.values():
                            s.step(total_loss / (step + 1))

                    if writer is not None and accelerator.is_main_process and global_step % config.training.log_steps == 0:
                        log_training_step(writer, loss, l_identity, l_emotion, l_landmark, l_recon, global_step, resolution)

                    if global_step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                        save_debug_images(x_s, x_t, outputs[0], outputs[1], global_step, resolution, config.training.output_dir)

            avg_loss = total_loss / len(train_dataloader)
            progress_bar.set_description(f"Res: {resolution}, Epoch {epoch+1}/{epochs_per_resolution}, Avg Loss: {avg_loss:.4f}")

            # Validation step
            val_loss = validate(model, val_dataloader, criterion, accelerator)
            print(f"Resolution {resolution}, Epoch {epoch+1}/{epochs_per_resolution}, Validation Loss: {val_loss:.4f}")

            # Update the schedulers with the validation loss after each epoch
            for s in schedulers.values():
                s.step(val_loss)

            # Save checkpoint
            if epoch % config.training.save_epochs == 0 and accelerator.is_main_process:
                save_path = os.path.join(config.training.output_dir, f"checkpoint-resolution-{resolution}-epoch-{epoch+1}")
                save_dict = {
                    'epoch': epoch,
                    'resolution': resolution,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'global_step': global_step
                }
                for k, opt in optimizers.items():
                    save_dict[f'optimizer_{k}_state_dict'] = opt.state_dict()
                accelerator.save(save_dict, save_path)

        # Reset start_epoch for the next resolution
        start_epoch = 0

    accelerator.end_training()




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

# writer, loss, l_identity, l_emotion, l_landmark, l_recon, global_step, resolution
def log_training_step(writer, loss, l_identity, l_emotion, l_landmark, l_recon, global_step, resolution):
    writer.add_scalar(f'Loss/Total/Resolution_{resolution}', loss, global_step)
    writer.add_scalar(f'Loss/Identity/Resolution_{resolution}', l_identity, global_step)
    writer.add_scalar(f'Loss/Emotion{resolution}', l_emotion, global_step)
    writer.add_scalar(f'Loss/landmark/Resolution_{resolution}', l_landmark, global_step)
    writer.add_scalar(f'Loss/Emotion/Resolution_{resolution}', l_emotion, global_step)
    writer.add_scalar(f'Loss/SelfReconstruction/Resolution_{resolution}', l_recon, global_step)



def validate(model, dataloader, criterion, accelerator):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x_s, x_t = batch["source_image"].to(accelerator.device), batch["target_image"].to(accelerator.device)
            emotion_labels_s, emotion_labels_t = batch["emotion_labels_s"].to(accelerator.device), batch["emotion_labels_t"].to(accelerator.device)

            outputs = model(x_s, x_t)
            loss, _, _, _, _, _,_ = criterion(x_s, x_t, *outputs, emotion_labels_s, emotion_labels_t)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

class GradientTracker:
    def __init__(self):
        self.last_valid_grads = {}

    def hook_fn(self, name):
        def hook(grad):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                # print(f"NaN in {name} - 🧹 cleaning..")
                return self.last_valid_grads.get(name, torch.zeros_like(grad))
            self.last_valid_grads[name] = grad.clone().detach()
            return grad
        return hook



def main():
    config = OmegaConf.load("config.yaml")
    
    # Create output and log directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    log_dir = os.path.join(config.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Tensorboard logs will be saved to: {log_dir}")

    # Set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1))  # Clamp values to [-1, 1]
    ])

    

    # Load the dataset
    # base_dataset = CelebADataset(config.dataset.name, config.dataset.split, preprocess)
    base_dataset = AffectNetDataset("/media/oem/12TB/AffectNet/train",  preprocess)

    # Initialize model, optimizer, and scheduler
    model = IRFD()
    optimizers = {
        'pose': AdamW(model.Ep.parameters(), lr=0.0001, weight_decay=0.01),
        'emotion': AdamW(model.Ee.parameters(), lr=0.0001, weight_decay=0.01),
        'identity': AdamW(model.Ei.parameters(), lr=0.0001, weight_decay=0.01),
        'other': AdamW([p for n, p in model.named_parameters() 
                        if not n.startswith(('Ep', 'Ee', 'Ei'))], lr=0.0001, weight_decay=0.01)
    }



    tracker = GradientTracker()

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(tracker.hook_fn(name))

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    progressive_irfd_train_loop(config, model, base_dataset, optimizers,  accelerator, writer, criterion, latest_checkpoint)

    # Close the SummaryWriter
    writer.close()



if __name__ == "__main__":
    main()


