import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

def load_and_preprocess_image(image_path, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def get_resnet_features(model, image_tensor):
    features = model(image_tensor)
    return features

def visualize_channel_activations_heatmap(features, num_channels=100):
    # Select a subset of channels to visualize
    activations = features[0, :num_channels].mean(dim=(1, 2)).cpu().detach().numpy()
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(activations.reshape(1, -1), cmap='viridis', annot=False, cbar=True)
    plt.title(f'Mean Activation of Top {num_channels} Channels')
    plt.xlabel('Channel Index')
    plt.show()

def visualize_feature_maps_grid(features, num_maps=16):
    # Select a subset of feature maps to visualize
    feature_maps = features[0, :num_maps].cpu().detach().numpy()
    
    # Create a grid to display the feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Map {i+1}')
    
    plt.tight_layout()
    plt.show()

def visualize_pca_reduction(features):
    # Reshape features to 2D
    features_2d = features.view(features.size(1), -1).t().cpu().detach().numpy()
    
    # Apply PCA
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_2d)
    
    # Normalize PCA results to [0, 1] for RGB visualization
    features_pca_norm = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    
    # Reshape back to image dimensions
    pca_image = features_pca_norm.reshape(7, 7, 3)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_image)
    plt.axis('off')
    plt.title('PCA Visualization of Features')
    plt.show()

def visualize_channel_correlation(features, num_channels=100):
    # Select a subset of channels
    selected_features = features[0, :num_channels].view(num_channels, -1).cpu().detach().numpy()
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(selected_features)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title(f'Channel Correlation Matrix (Top {num_channels} Channels)')
    plt.show()

def main():
    # Load pre-trained ResNet model
    model = resnet50(pretrained=True)
    features_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    features_extractor.eval()

    # Load and preprocess an image
    image_path = 'S.png'  # Replace with your image path
    image_tensor = load_and_preprocess_image(image_path)

    # Get features
    with torch.no_grad():
        features = features_extractor(image_tensor)

    print(f"Feature shape: {features.shape}")

    # Visualize channel activations as a heatmap
    visualize_channel_activations_heatmap(features)

    # Visualize a subset of feature maps
    visualize_feature_maps_grid(features)

    # Visualize PCA reduction
    visualize_pca_reduction(features)

    # Visualize channel correlation
    visualize_channel_correlation(features)

if __name__ == "__main__":
    main()