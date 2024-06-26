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

class IRFD(nn.Module):
    def __init__(self):
        super(IRFD, self).__init__()
        
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        self.Gd = self._create_generator()
        
        # Emotion classifier
        self.Cm = nn.Linear(2048, 8)  # Assuming 8 emotion categories
        
    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def _create_generator(self):
        # Simplified generator structure
        return nn.Sequential(
            nn.Linear(2048 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 256 * 3),
            nn.Tanh()
        )
    
    def forward(self, x_s, x_t):
        # Encode source and target images
        fi_s = self.Ei(x_s)
        fe_s = self.Ee(x_s)
        fp_s = self.Ep(x_s)
        
        fi_t = self.Ei(x_t)
        fe_t = self.Ee(x_t)
        fp_t = self.Ep(x_t)
        
        # Randomly swap one type of feature
        swap_type = torch.randint(0, 3, (1,)).item()
        if swap_type == 0:
            fi_s, fi_t = fi_t, fi_s
        elif swap_type == 1:
            fe_s, fe_t = fe_t, fe_s
        else:
            fp_s, fp_t = fp_t, fp_s
        
        # Generate reconstructed images
        x_s_recon = self.Gd(torch.cat([fi_s, fe_s, fp_s], dim=1))
        x_t_recon = self.Gd(torch.cat([fi_t, fe_t, fp_t], dim=1))
        
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t

class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels):
        # Identity loss
        l_identity = torch.max(
            self.l2_loss(fi_s, fi_t) - self.l2_loss(fi_s, fi_s) + self.alpha,
            torch.tensor(0.0).to(fi_s.device)
        )
        
        # Classification loss
        l_cls = self.ce_loss(fe_s, emotion_labels) + self.ce_loss(fe_t, emotion_labels)
        
        # Pose loss
        l_pose = self.l2_loss(fp_s, fp_t)
        
        # Emotion loss
        l_emotion = self.l2_loss(fe_s, fe_t)
        
        # Self-reconstruction loss
        l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self
        
        return total_loss


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
            images = batch["pixel_values"].to(accelerator.device)
            x_s, x_t, emotion_labels = batch
            x_s, x_t, emotion_labels = x_s.to(accelerator.device), x_t.to(accelerator.device), emotion_labels.to(accelerator.device)

            with accelerator.accumulate(model):
                x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t = model(x_s, x_t)
                loss = criterion(x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels)

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
    
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"pixel_values": images}

    dataset.set_transform(transform)
    dataloader = DataLoader(dataset, batch_size=config.training.train_batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.optimization.learning_rate)

    train_loop(config, model, dataloader, optimizer)

if __name__ == "__main__":
    main()