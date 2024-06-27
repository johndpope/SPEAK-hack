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


def train_loop(config, model, dataloader, optimizer, accelerator, start_epoch=0, global_step=0, writer=None):

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
            
            emotion_labels_s = batch["emotion_labels_s"]
            emotion_labels_t = batch["emotion_labels_t"]

            with accelerator.accumulate(model):
                x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t = model(x_s, x_t)
          
                loss = criterion(x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                
                # Log to tensorboard
                if writer is not None and accelerator.is_main_process:
                    writer.add_scalar('Loss/train', logs['loss'], global_step)

                # The accelerator.log() call can be kept for compatibility
                accelerator.log(logs, step=global_step)


            if global_step % config.training.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            # Save reconstructed image periodically
            if global_step % config.training.save_image_steps == 0 and accelerator.is_main_process:
                # Denormalize the image
                x_s_recon_denorm = x_s_recon * 0.5 + 0.5
                save_image(x_s_recon_denorm[0], os.path.join(recon_dir, f"recon_step_{global_step}.png"))


    accelerator.end_training()

def main():
    config = OmegaConf.load("config.yaml")

  # Create the output directory if it doesn't exist
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Set up logging directory
    log_dir = os.path.join(config.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Tensorboard logs will be saved to: {log_dir}")


    model = IRFD()

    dataset = load_dataset(config.dataset.name, split=config.dataset.split)
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    

    fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
    
    emotion_class_to_idx = {
        'angry': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 
        'neutral': 5, 'sad': 6, 'surprise': 7
    }
    def transform(examples):
        source_images = []
        target_images = []
        emotion_labels_s_list = []
        emotion_labels_t_list = []

        for i in range(0, len(examples["image"]), 2):
            source_image = preprocess(examples["image"][i].convert("RGB"))
            target_image = preprocess(examples["image"][i + 1].convert("RGB"))

            source_image_np = source_image.permute(1, 2, 0).numpy()
            target_image_np = target_image.permute(1, 2, 0).numpy()

            emotion_labels_s = fer.predict_emotions(source_image_np, logits=False)[0].lower()
            emotion_labels_t = fer.predict_emotions(target_image_np, logits=False)[0].lower()

            source_images.append(source_image)
            target_images.append(target_image)
            emotion_labels_s_list.append(emotion_class_to_idx[emotion_labels_s])
            emotion_labels_t_list.append(emotion_class_to_idx[emotion_labels_t])

        emotion_labels_s = torch.tensor(emotion_labels_s_list, dtype=torch.long)
        emotion_labels_t = torch.tensor(emotion_labels_t_list, dtype=torch.long)

        return {
            "source_image": source_images,
            "emotion_labels_s": emotion_labels_s,
            "target_image": target_images,
            "emotion_labels_t": emotion_labels_t
        }
    dataset.set_transform(transform)
    dataloader = DataLoader(dataset, batch_size=config.training.train_batch_size, shuffle=True)

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

    train_loop(config, model, dataloader, optimizer, accelerator,start_epoch, global_step)


if __name__ == "__main__":
    main()