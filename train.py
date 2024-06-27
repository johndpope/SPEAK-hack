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


def train_loop(config, model, dataloader, optimizer):
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    criterion = IRFDLoss().to(accelerator.device)

    global_step = 0

    for epoch in range(config.training.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            x_s = batch["source_image"].to(accelerator.device)
            x_t = batch["target_image"].to(accelerator.device)
            
            x_s = x_s.unsqueeze(0)
            x_t = x_t.unsqueeze(0)
            
            print("x_s:",x_s.shape)
            print("x_t:",x_t.shape)
        
            emotion_labels_t = model.fer.predict_emotions(x_t.cpu().numpy())
            print("emotion_labels_s:",emotion_labels_s)
            print("emotion_labels_t:",emotion_labels_t)
        
            with accelerator.accumulate(model):
                x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t = model(x_s, x_t)
                print("x_s_recon:",x_s_recon.shape)
                print("fi_s:",fi_s.shape)
                print("fe_s:",fe_s.shape)
                print("fp_s:",fp_s.shape)

                # Convert emotion labels to indices
                emotion_labels_s = torch.tensor([model.emotion_idx_to_class[label] for label in emotion_labels_s], dtype=torch.long)
                emotion_labels_t = torch.tensor([model.emotion_idx_to_class[label] for label in emotion_labels_t], dtype=torch.long)
                
                loss = criterion(x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels_s, emotion_labels_t)

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
                accelerator.log(logs, step=global_step)

            if global_step % config.training.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

    accelerator.end_training()

def main():
    config = OmegaConf.load("config.yaml")

    model = IRFD()

    dataset = load_dataset(config.dataset.name, split=config.dataset.split)
    preprocess = transforms.Compose([
        transforms.Resize((config.model.sample_size, config.model.sample_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    

    fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
    
    def transform(examples):
        image_pairs = []
        for i in range(0, len(examples["image"]), 2):
            source_image = preprocess(examples["image"][i].convert("RGB"))
            target_image = preprocess(examples["image"][i + 1].convert("RGB"))

            source_image_np = source_image.permute(1, 2, 0).numpy()
            target_image_np = target_image.permute(1, 2, 0).numpy()

            emotion_labels_s = model.fer.predict_emotions(source_image_np)
            emotion_labels_t = model.fer.predict_emotions(target_image_np)

        return {"source_image":source_image,"emotion_labels_s":emotion_labels_s,"target_image":target_image,"emotion_labels_t":emotion_labels_t}

    dataset.set_transform(transform)
    dataloader = DataLoader(dataset, batch_size=config.training.train_batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.optimization.learning_rate)

    train_loop(config, model, dataloader, optimizer)

if __name__ == "__main__":
    main()