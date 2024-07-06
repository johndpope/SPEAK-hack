import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from rembg import remove
from PIL import Image
import io
import numpy as np
import random


import os


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
from rembg import remove
import hashlib

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, preprocess, remove_background=False, use_greenscreen=False, cache_dir=None):
        self.root_dir = os.path.join(root_dir)
        self.preprocess = preprocess
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define emotion class to index mapping
        self.emotion_class_to_idx = {str(i): i for i in range(8)}

        # Load image paths and their corresponding emotion labels
        self.image_paths = []
        self.emotion_labels = []
        for emotion_label in range(8):
            emotion_dir = os.path.join(self.root_dir, str(emotion_label))
            for img_file in os.listdir(emotion_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(emotion_dir, img_file))
                    self.emotion_labels.append(emotion_label)
    
    def __len__(self):
        return len(self.image_paths) // 2  # We're processing pairs of images

    def get_cache_path(self, image_path):
        if not self.cache_dir:
            return None
        image_hash = hashlib.md5(image_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{image_hash}.png")

    def check_image_quality(self, image):
        # Check if the image is mostly transparent or black
        np_image = np.array(image)
        if np_image.shape[2] == 4:  # RGBA
            non_transparent = np_image[:,:,3].sum()
            total_pixels = np_image.shape[0] * np_image.shape[1]
            if non_transparent / total_pixels < 0.1:  # Less than 10% non-transparent
                return False
        
        # Check if the image is too dark
        gray_image = np.array(image.convert('L'))
        if np.mean(gray_image) < 30:  # Average pixel value less than 30 (out of 255)
            return False
        
        return True
        
    def remove_bg(self, image, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            bg_removed_image = Image.open(cache_path).convert("RGB")
            if self.check_image_quality(bg_removed_image):
                return bg_removed_image
            else:
                print(f"Cached background-removed image failed quality check. Using original image.")
                return image.convert("RGB")

        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            bg_removed_bytes = remove(img_byte_arr)
            bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")

            if self.use_greenscreen:
                black_screen = Image.new("RGB", bg_removed_image.size, (0, 0, 0))
                final_image = Image.alpha_composite(black_screen.convert("RGBA"), bg_removed_image)
            else:
                final_image = bg_removed_image

            final_image = final_image.convert("RGB")

            if self.check_image_quality(final_image):
                if cache_path:
                    final_image.save(cache_path)
                return final_image
            else:
                print(f"Background-removed image failed quality check. Using original image.")
                return image.convert("RGB")

        except Exception as e:
            print(f"Background removal failed: {str(e)}. Returning original image.")
            return image.convert("RGB")

    def __getitem__(self, idx):
        # Get a pair of images
        source_idx = idx * 2
        target_idx = source_idx + 1

        # Process source image
        source_image_path = self.image_paths[source_idx]
        source_cache_path = self.get_cache_path(source_image_path) if self.remove_background else None
        source_image = Image.open(source_image_path).convert("RGB")
        if self.remove_background:
            source_image = self.remove_bg(source_image, source_cache_path)
        source_image = self.preprocess(source_image)
        source_emotion_idx = self.emotion_labels[source_idx]

        # Process target image
        target_image_path = self.image_paths[target_idx]
        target_cache_path = self.get_cache_path(target_image_path) if self.remove_background else None
        target_image = Image.open(target_image_path).convert("RGB")
        if self.remove_background:
            target_image = self.remove_bg(target_image, target_cache_path)
        target_image = self.preprocess(target_image)
        target_emotion_idx = self.emotion_labels[target_idx]

        return {
            "source_image": source_image,
            "target_image": target_image,
            "emotion_labels_s": torch.tensor(source_emotion_idx, dtype=torch.long),
            "emotion_labels_t": torch.tensor(target_emotion_idx, dtype=torch.long)
        }



class CelebADataset(Dataset):
    def __init__(self, dataset_name, split, preprocess, remove_background=False, use_greenscreen=False):
        self.dataset = load_dataset(dataset_name, split=split)
        self.preprocess = preprocess
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
        self.emotion_class_to_idx = {
            'angry': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 
            'neutral': 5, 'sad': 6, 'surprise': 7
        }
        self.remove_background = False
        self.use_greenscreen = False

    def __len__(self):
        return len(self.dataset) // 2  # We're processing pairs of images

    def remove_bg(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")  # Use RGBA to keep transparency

        if self.use_greenscreen:
            # Create a green screen background
            # green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))  # Green color
            black_screen = Image.new("RGB", bg_removed_image.size, (0, 0, 0))
            # Composite the image onto the green screen
            final_image = Image.alpha_composite(black_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        final_image = final_image.convert("RGB")  # Convert to RGB format
        return final_image

    def __getitem__(self, idx):
        # Get a pair of images
        source_idx = idx * 2
        target_idx = source_idx + 1

        # Process source image
        source_image = self.dataset[source_idx]['image'].convert("RGB")
        if self.remove_background:
            source_image = self.remove_bg(source_image)
        source_image = self.preprocess(source_image)
        source_image_np = source_image.permute(1, 2, 0).numpy()
        source_emotion = self.fer.predict_emotions(source_image_np, logits=False)[0].lower()
        source_emotion_idx = self.emotion_class_to_idx[source_emotion]

        # Process target image
        target_image = self.dataset[target_idx]['image'].convert("RGB")
        if self.remove_background:
            target_image = self.remove_bg(target_image)
        target_image = self.preprocess(target_image)
        target_image_np = target_image.permute(1, 2, 0).numpy()
        target_emotion = self.fer.predict_emotions(target_image_np, logits=False)[0].lower()
        target_emotion_idx = self.emotion_class_to_idx[target_emotion]

        return {
            "source_image": source_image,
            "target_image": target_image,
            "emotion_labels_s": torch.tensor(source_emotion_idx, dtype=torch.long),
            "emotion_labels_t": torch.tensor(target_emotion_idx, dtype=torch.long)
        }

class ProgressiveDataset(Dataset):
    def __init__(self, base_dataset, current_resolution):
        self.base_dataset = base_dataset
        self.current_resolution = current_resolution
        self.resize_transform = transforms.Resize(current_resolution)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Resize images to current resolution
        item["source_image"] = self.resize_transform(item["source_image"].unsqueeze(0)).squeeze(0)
        item["target_image"] = self.resize_transform(item["target_image"].unsqueeze(0)).squeeze(0)
        
        return item



class CelebADatasetWithAugmentation(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Increase brightness and contrast
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        
        self.cutout = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        augmented_image = self.transform(image)
        
        # Apply cutout
        if random.random() < 0.5:
            augmented_image = self.cutout(augmented_image)
        
        return augmented_image, label


class OverfitDataset(Dataset):
    def __init__(self, source_image_path, target_image_path, preprocess=None):
        self.source_image = Image.open(source_image_path).convert("RGB")
        self.target_image = Image.open(target_image_path).convert("RGB")
        
        if preprocess is None:
            self.preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.preprocess = preprocess

    def __len__(self):
        return 1  # We only have one pair of images

    def __getitem__(self, idx):
        source_image = self.preprocess(self.source_image)
        target_image = self.preprocess(self.target_image)

        return {
            "source_image": source_image,
            "target_image": target_image,
            "emotion_labels_s": torch.tensor(0, dtype=torch.long),  # Placeholder emotion label
            "emotion_labels_t": torch.tensor(0, dtype=torch.long)   # Placeholder emotion label
        }

class ProgressiveOverfitDataset(Dataset):
    def __init__(self, base_dataset, current_resolution):
        self.base_dataset = base_dataset
        self.current_resolution = current_resolution
        self.resize_transform = transforms.Resize(current_resolution)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Resize images to current resolution
        item["source_image"] = self.resize_transform(item["source_image"].unsqueeze(0)).squeeze(0)
        item["target_image"] = self.resize_transform(item["target_image"].unsqueeze(0)).squeeze(0)
        
        return item