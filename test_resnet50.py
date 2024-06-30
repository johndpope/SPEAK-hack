import torch
from torchvision import models, transforms
from PIL import Image
import sys

def test_resnet50_feature_extraction(image_path):
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set to evaluation mode

    # Remove the final fully-connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print(f"Error: Unable to open image at {image_path}")
        return

    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Extract features
    with torch.no_grad():
        try:
            features = model(img_tensor)
            print("Feature extraction successful!")
            print(f"Feature shape: {features.shape}")
            print(f"Feature statistics:")
            print(f"  Min: {features.min().item():.4f}")
            print(f"  Max: {features.max().item():.4f}")
            print(f"  Mean: {features.mean().item():.4f}")
            print(f"  Std: {features.std().item():.4f}")
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")

if __name__ == "__main__":

        image_path = '/media/oem/12TB/SPEAK-hack/speak/x_s_failed.png'
        test_resnet50_feature_extraction(image_path)

        image_path = '/media/oem/12TB/SPEAK-hack/speak/x_t_failed.png'
        test_resnet50_feature_extraction(image_path)