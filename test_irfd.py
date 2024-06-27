import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import IRFD
import os
from accelerate import Accelerator
from omegaconf import OmegaConf

def test_irfd(config, test_image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize IRFD model
    irfd = IRFD().to(device)

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.training.output_dir, "logs"),
    )
    
    # Find the latest checkpoint
    latest_checkpoint = None
    if os.path.exists(config.training.output_dir):
        checkpoints = [d for d in os.listdir(config.training.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))

    if latest_checkpoint:
        checkpoint_path = os.path.join(config.training.output_dir, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        global_step = int(latest_checkpoint.split("-")[1])
        print(f"Resuming from global step {global_step}")
    else:
        print("No checkpoint found. Starting from scratch.")
        return

    # Prepare the model with Accelerator
    irfd = accelerator.prepare(irfd)
    irfd.eval()
    


    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load and preprocess test images
    test_images = []
    for path in test_image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        test_images.append(img_tensor)

    with torch.no_grad():
        # Extract features
        identity_features = []
        emotion_features = []
        pose_features = []

        for img in test_images:
            fi = irfd.Ei(img)
            fe = irfd.Ee(img)
            fp = irfd.Ep(img)
            identity_features.append(fi)
            emotion_features.append(fe)
            pose_features.append(fp)

        # Test disentanglement by swapping features
        num_images = len(test_images)
        reconstructed_images = []

        for i in range(num_images):
            for j in range(num_images):
                for k in range(num_images):
                    # Combine features from different images
                    combined_features = torch.cat([
                        identity_features[i],
                        emotion_features[j],
                        pose_features[k]
                    ], dim=1)

                    # Reconstruct image
                    reconstructed = irfd.Gd(combined_features)
                    reconstructed_images.append(reconstructed.squeeze(0))

        # Visualize results
        fig, axs = plt.subplots(num_images, num_images * num_images, figsize=(20, 5 * num_images))
        
        for i, img in enumerate(reconstructed_images):
            row = i // (num_images * num_images)
            col = i % (num_images * num_images)
            axs[row, col].imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            axs[row, col].axis('off')
            
            if col == 0:
                axs[row, col].set_title(f"ID: {i//(num_images*num_images) + 1}")
            if row == 0:
                axs[row, col].set_title(f"Emotion: {(i//num_images) % num_images + 1}\nPose: {i % num_images + 1}", pad=20)

        plt.tight_layout()
        plt.savefig('irfd_test_results.png')
        plt.close()

        print("IRFD test completed. Results saved as 'irfd_test_results.png'")

def main():
    config = OmegaConf.load("config.yaml")
    test_image_paths = [
        "S.png",
        "T.png"
    ]
    test_irfd(config, test_image_paths)
if __name__ == "__main__":
    main()