import torch
from torchvision.models import resnet50

def test_resnet50_feature_size():
    # Create a ResNet50 model
    model = resnet50(pretrained=True)
    
    # Remove the average pooling and fully connected layers
    features = torch.nn.Sequential(*list(model.children())[:-2])
    
    # Set the model to evaluation mode
    features.eval()
    
    # Create a sample input
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    
    # Forward pass
    with torch.no_grad():
        output = features(sample_input)
    
    print(f"ResNet50 feature output shape: {output.shape}")
    print(f"Number of channels: {output.shape[1]}")
    print(f"Spatial dimensions: {output.shape[2]}x{output.shape[3]}")

if __name__ == "__main__":
    test_resnet50_feature_size()