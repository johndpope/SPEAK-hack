import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

class CelebADataset(Dataset):
    def __init__(self, dataset_name, split, preprocess):
        self.dataset = load_dataset(dataset_name, split=split)
        self.preprocess = preprocess
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
        self.emotion_class_to_idx = {
            'angry': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 
            'neutral': 5, 'sad': 6, 'surprise': 7
        }

    def __len__(self):
        return len(self.dataset) // 2  # We're processing pairs of images

    def __getitem__(self, idx):
        # Get a pair of images
        source_idx = idx * 2
        target_idx = source_idx + 1

        # Process source image
        source_image = self.preprocess(self.dataset[source_idx]['image'].convert("RGB"))
        source_image_np = source_image.permute(1, 2, 0).numpy()
        source_emotion = self.fer.predict_emotions(source_image_np, logits=False)[0].lower()
        source_emotion_idx = self.emotion_class_to_idx[source_emotion]

        # Process target image
        target_image = self.preprocess(self.dataset[target_idx]['image'].convert("RGB"))
        target_image_np = target_image.permute(1, 2, 0).numpy()
        target_emotion = self.fer.predict_emotions(target_image_np, logits=False)[0].lower()
        target_emotion_idx = self.emotion_class_to_idx[target_emotion]

        return {
            "source_image": source_image,
            "target_image": target_image,
            "emotion_labels_s": torch.tensor(source_emotion_idx, dtype=torch.long),
            "emotion_labels_t": torch.tensor(target_emotion_idx, dtype=torch.long)
        }


class ProgressiveCelebADataset(Dataset):
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
