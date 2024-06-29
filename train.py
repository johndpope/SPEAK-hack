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
from CelebADataset import CelebADataset
from MRLR import IRFDWithMRLR


def save_debug_images(x_s, x_t, x_s_recon, x_t_recon, step, output_dir):
    # Denormalize images
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    x_s = denorm(x_s)
    x_t = denorm(x_t)
    x_s_recon = denorm(x_s_recon)
    x_t_recon = denorm(x_t_recon)
    
    # Concatenate input and reconstructed images
    combined = torch.cat([x_s, x_s_recon, x_t, x_t_recon], dim=0)
    
    # Save up to 16 sets of images (64 images total)
    num_sets = min(16, x_s.size(0))
    save_image(combined[:num_sets*4], os.path.join(output_dir, f"debug_step_{step}.png"), nrow=4)



def train_loop(config, model, dataloader, optimizer, accelerator, writer=None,start_epoch=0, global_step=0 ):

    # Create a directory to save reconstructed images
    recon_dir = os.path.join(config.training.output_dir, "output_images")
    if accelerator.is_main_process:
        os.makedirs(recon_dir, exist_ok=True)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    criterion = IRFDLoss().to(accelerator.device)

    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            x_s = batch["source_image"].to(accelerator.device)
            x_t = batch["target_image"].to(accelerator.device)
            
            emotion_labels_s = batch["emotion_labels_s"].to(accelerator.device)
            emotion_labels_t = batch["emotion_labels_t"].to(accelerator.device)

    

            with accelerator.accumulate(model):
                x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t = model(x_s, x_t)
                loss, l_identity, l_cls, l_pose, l_emotion, l_self = criterion(x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t)

    
                x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t = model(x_s, x_t)          
                loss, l_identity, l_cls, l_pose, l_emotion, l_self = criterion(x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "total_loss": loss.detach().item(),
                    "identity_loss": l_identity.detach().item(),
                    "classification_loss": l_cls.detach().item(),
                    "pose_loss": l_pose.detach().item(),
                    "emotion_loss": l_emotion.detach().item(),
                    "self_reconstruction_loss": l_self.detach().item(),
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                
                # Log to tensorboard
                if writer is not None and accelerator.is_main_process:
                    writer.add_scalar('Loss/Total', logs['total_loss'], global_step)
                    writer.add_scalar('Loss/Identity', logs['identity_loss'], global_step)
                    writer.add_scalar('Loss/Classification', logs['classification_loss'], global_step)
                    writer.add_scalar('Loss/Pose', logs['pose_loss'], global_step)
                    writer.add_scalar('Loss/Emotion', logs['emotion_loss'], global_step)
                    writer.add_scalar('Loss/SelfReconstruction', logs['self_reconstruction_loss'], global_step)

                # The accelerator.log() call can be kept for compatibility
                accelerator.log(logs, step=global_step)


            if global_step % config.training.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            # Save reconstructed image periodically
            if global_step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                save_debug_images(x_s, x_t, x_s_recon, x_t_recon, global_step, recon_dir)
                

    accelerator.end_training()

def main():
    config = OmegaConf.load("config.yaml")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # "lansinuote/gen.1.celeba" from huggingface
    dataset = CelebADataset(config.dataset.name, config.dataset.split, preprocess)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
  
  # Create the output directory if it doesn't exist
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Set up logging directory
    log_dir = os.path.join(config.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Tensorboard logs will be saved to: {log_dir}")


    # model = IRFD()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRFDWithMRLR(device)
    optimizer = optim.Adam(model.parameters(), lr=config.optimization.learning_rate)
  
  # Check if a checkpoint exists
    latest_checkpoint = None
    if os.path.exists(config.training.output_dir):
        checkpoints = [d for d in os.listdir(config.training.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))

    start_epoch = 0
    global_step = 0
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )
    # Initialize tensorboard SummaryWriter
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)


    if latest_checkpoint:
        checkpoint_path = os.path.join(config.training.output_dir, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        global_step = int(latest_checkpoint.split("-")[1])
        start_epoch = global_step // len(dataloader)  # Approximate the starting epoch

    train_loop(config, model, dataloader, optimizer, accelerator,writer,start_epoch, global_step)


if __name__ == "__main__":
    main()