#!/usr/bin/env python3
"""
Corrected Building Segmentation Model with Proper Ground Truth Generation

This script correctly generates ground truth masks by overlaying building boundaries
from labels onto images, then trains a U-Net model for building segmentation.

Author: AI Assistant
Date: 2024
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any
from shapely.geometry import Polygon
from shapely import wkt
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedBuildingDataset(Dataset):
    """
    Dataset class for building segmentation with corrected ground truth generation.
    """
    
    def __init__(self, image_paths: List[str], label_paths: List[str], 
                 transform=None, is_training=True):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            label_paths: List of paths to label JSON files
            transform: Transform pipeline
            is_training: Whether this is training data
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Load labels and create mask
        label_path = self.label_paths[idx]
        mask = self.create_corrected_mask_from_json(label_path, original_shape)
        
        # Resize both image and mask to standard size
        image_resized = cv2.resize(image, (512, 512))
        mask_resized = cv2.resize(mask, (512, 512))
        
        # Apply transforms
        if self.transform:
            # Convert to PIL for transforms
            image_pil = Image.fromarray(image_resized)
            mask_pil = Image.fromarray(mask_resized)
            
            # Apply transforms
            if hasattr(self.transform, '__call__'):
                # Custom transform function
                image_tensor, mask_tensor = self.transform(image_pil, mask_pil)
            else:
                # Standard torch transforms
                image_tensor = self.transform(image_pil)
                mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        else:
            # Convert to tensors
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_resized).float() / 255.0
        
        return image_tensor, mask_tensor
    
    def create_corrected_mask_from_json(self, json_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a corrected binary mask from JSON label file by properly overlaying
        building boundaries onto the image.
        
        Args:
            json_path: Path to JSON label file
            image_shape: Shape of the image (height, width)
            
        Returns:
            Binary mask array
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Use xy coordinates for pixel-level segmentation
            features = data.get('features', {}).get('xy', [])
            
            for feature in features:
                if feature.get('properties', {}).get('feature_type') == 'building':
                    wkt_string = feature.get('wkt', '')
                    if wkt_string:
                        try:
                            # Parse WKT to get polygon coordinates
                            polygon = wkt.loads(wkt_string)
                            coords = list(polygon.exterior.coords)
                            
                            # Convert to integer coordinates and ensure they're within image bounds
                            points = []
                            for coord in coords:
                                x, y = int(coord[0]), int(coord[1])
                                # Clamp coordinates to image bounds
                                x = max(0, min(x, image_shape[1] - 1))
                                y = max(0, min(y, image_shape[0] - 1))
                                points.append([x, y])
                            
                            if len(points) >= 3:  # Need at least 3 points for a polygon
                                points_array = np.array(points, dtype=np.int32)
                                
                                # Fill the polygon in the mask
                                cv2.fillPoly(mask, [points_array], 255)
                                
                        except Exception as e:
                            logger.warning(f"Error processing polygon in {json_path}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
        
        return mask


class UNet(nn.Module):
    """
    U-Net architecture for building segmentation.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final layer
        output = self.final(dec1)
        
        return torch.sigmoid(output)


class CorrectedBuildingSegmentationModel:
    """
    Main class for corrected building segmentation model training and evaluation.
    """
    
    def __init__(self, model_path: str = "corrected_building_segmentation_model.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.optimizer = None
        self.criterion = None
        
        logger.info(f"Using device: {self.device}")
        
    def setup_training(self, learning_rate=1e-4):
        """Setup optimizer and loss function for training."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)  # Add channel dimension
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation", leave=False)
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50):
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        # Calculate estimated time
        start_time = time.time()
        
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        for epoch in epoch_pbar:
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            avg_epoch_time = total_time / (epoch + 1)
            remaining_time = avg_epoch_time * (epochs - epoch - 1)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'epoch_time': f'{epoch_time:.1f}s',
                'eta': f'{remaining_time/60:.1f}m'
            })
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Save best model
            if epoch == 0 or val_loss < min(val_losses[:-1]):
                self.save_model()
                logger.info("New best model saved!")
        
        total_training_time = time.time() - start_time
        logger.info(f"Training completed in {total_training_time/60:.1f} minutes ({total_training_time:.1f} seconds)")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
    def save_model(self):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the model."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {self.model_path}")
            return True
        return False
    
    def predict(self, image_path: str) -> np.ndarray:
        """Predict building segmentation for a single image."""
        self.model.eval()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Resize to model input size
        image_resized = cv2.resize(image, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.squeeze().cpu().numpy()
        
        # Resize back to original size
        prediction_resized = cv2.resize(prediction, (original_shape[1], original_shape[0]))
        
        return prediction_resized
    
    def calculate_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics. Both masks are resized to 512x512 for consistency.
        """
        # Resize both masks to 512x512
        pred_mask_resized = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_mask_resized = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Convert to binary
        pred_binary = (pred_mask_resized > 0.5).astype(np.uint8)
        gt_binary = (gt_mask_resized > 0.5).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()

        # IoU
        iou = intersection / union if union > 0 else 0.0

        # Precision, Recall, F1
        tp = intersection
        fp = np.logical_and(pred_binary, 1 - gt_binary).sum()
        fn = np.logical_and(1 - pred_binary, gt_binary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy
        accuracy = (pred_binary == gt_binary).sum() / pred_binary.size

        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def create_visualization(self, image_path: str, gt_mask: np.ndarray, pred_mask: np.ndarray, 
               metrics: Dict[str, float], save_path: str):
    
        # Create visualization showing input image, ground truth, and prediction. All masks and images are resized to 512x512.
        # Load and resize original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        # Resize masks to 512x512
        gt_mask_resized = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_mask_resized = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Create overlays
        gt_overlay = image.copy()
        pred_overlay = image.copy()

        # Ground truth overlay (green)
        gt_binary = (gt_mask_resized > 0.5).astype(np.uint8)
        gt_overlay[gt_binary > 0] = [0, 255, 0]  # Green for ground truth

        # Prediction overlay (red)
        pred_binary = (pred_mask_resized > 0.5).astype(np.uint8)
        pred_overlay[pred_binary > 0] = [255, 0, 0]  # Red for prediction

        # Create combined overlay
        combined_overlay = image.copy()
        combined_overlay[gt_binary > 0] = [0, 255, 0]  # Green for ground truth
        combined_overlay[pred_binary > 0] = [255, 0, 0]  # Red for prediction

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Ground truth
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Prediction
        axes[1, 0].imshow(pred_overlay)
        axes[1, 0].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Combined overlay
        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Combined (GT: Green, Pred: Red)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Add metrics text
        metrics_text = f"""
        IoU: {metrics['iou']:.3f}
        Precision: {metrics['precision']:.3f}
        Recall: {metrics['recall']:.3f}
        F1: {metrics['f1']:.3f}
        Accuracy: {metrics['accuracy']:.3f}
        """

        fig.text(0.02, 0.02, metrics_text, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {save_path}")
        
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """Plot training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png')
        plt.close()


def create_data_loaders(images_dir: str, labels_dir: str, batch_size: int = 4, 
                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for pre-disaster images only.
    """
    # Get pre-disaster images only
    image_files = list(Path(images_dir).glob("*.png"))
    predisaster_images = [img for img in image_files if "pre_disaster" in img.name]
    
    logger.info(f"Found {len(predisaster_images)} pre-disaster images")
    
    # Get corresponding label files
    label_paths = []
    valid_image_paths = []
    
    for image_path in predisaster_images:
        label_path = Path(labels_dir) / f"{image_path.stem}.json"
        if label_path.exists():
            label_paths.append(str(label_path))
            valid_image_paths.append(str(image_path))
        else:
            logger.warning(f"Label file not found for {image_path.name}")
    
    logger.info(f"Found {len(valid_image_paths)} valid image-label pairs")
    
    # Split data
    train_images, val_images, train_labels, val_labels = train_test_split(
        valid_image_paths, label_paths, train_size=train_split, random_state=42
    )
    
    # Create datasets
    train_dataset = CorrectedBuildingDataset(train_images, train_labels, transform=None, is_training=True)
    val_dataset = CorrectedBuildingDataset(val_images, val_labels, transform=None, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def evaluate_model(model, images_dir: str, labels_dir: str, max_images: int = 10):
    """
    Evaluate the trained model on pre-disaster images.
    """
    # Get pre-disaster images
    image_files = list(Path(images_dir).glob("*.png"))
    predisaster_images = [img for img in image_files if "pre_disaster" in img.name][:max_images]
    
    logger.info(f"Evaluating {len(predisaster_images)} pre-disaster images...")
    
    # Create output directory
    output_dir = "corrected_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics storage
    metrics = {
        'iou_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': [],
        'accuracy_scores': [],
        'image_names': []
    }
    
    # Process each image
    eval_pbar = tqdm(enumerate(predisaster_images), total=len(predisaster_images), desc="Evaluating Images")
    for i, image_path in eval_pbar:
        eval_pbar.set_description(f"Evaluating {image_path.name}")
        
        # Get corresponding label file
        label_path = Path(labels_dir) / f"{image_path.stem}.json"
        
        if not label_path.exists():
            logger.warning(f"Label file not found for {image_path.name}")
            continue
        
        # Load ground truth mask
        dataset = CorrectedBuildingDataset([str(image_path)], [str(label_path)])
        _, gt_mask = dataset[0]
        gt_mask = gt_mask.numpy()
        
        # Get prediction
        pred_mask = model.predict(str(image_path))
        
        # Calculate metrics
        image_metrics = model.calculate_metrics(pred_mask, gt_mask)
        
        # Store metrics
        metrics['iou_scores'].append(image_metrics['iou'])
        metrics['precision_scores'].append(image_metrics['precision'])
        metrics['recall_scores'].append(image_metrics['recall'])
        metrics['f1_scores'].append(image_metrics['f1'])
        metrics['accuracy_scores'].append(image_metrics['accuracy'])
        metrics['image_names'].append(image_path.name)
        
        # Update progress bar with current metrics
        eval_pbar.set_postfix({
            'IoU': f'{image_metrics["iou"]:.3f}',
            'F1': f'{image_metrics["f1"]:.3f}'
        })
        
        # Create visualization
        viz_path = Path(output_dir) / f"{image_path.stem}_evaluation.png"
        model.create_visualization(str(image_path), gt_mask, pred_mask, image_metrics, str(viz_path))
    
    # Calculate overall metrics
    overall_metrics = {
        'mean_iou': np.mean(metrics['iou_scores']),
        'std_iou': np.std(metrics['iou_scores']),
        'mean_precision': np.mean(metrics['precision_scores']),
        'std_precision': np.std(metrics['precision_scores']),
        'mean_recall': np.mean(metrics['recall_scores']),
        'std_recall': np.std(metrics['recall_scores']),
        'mean_f1': np.mean(metrics['f1_scores']),
        'std_f1': np.std(metrics['f1_scores']),
        'mean_accuracy': np.mean(metrics['accuracy_scores']),
        'std_accuracy': np.std(metrics['accuracy_scores']),
        'total_images': len(metrics['iou_scores'])
    }
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("CORRECTED MODEL EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total Images Evaluated: {overall_metrics['total_images']}")
    logger.info(f"Mean IoU: {overall_metrics['mean_iou']:.3f} ± {overall_metrics['std_iou']:.3f}")
    logger.info(f"Mean Precision: {overall_metrics['mean_precision']:.3f} ± {overall_metrics['std_precision']:.3f}")
    logger.info(f"Mean Recall: {overall_metrics['mean_recall']:.3f} ± {overall_metrics['std_recall']:.3f}")
    logger.info(f"Mean F1: {overall_metrics['mean_f1']:.3f} ± {overall_metrics['std_f1']:.3f}")
    logger.info(f"Mean Accuracy: {overall_metrics['mean_accuracy']:.3f} ± {overall_metrics['std_accuracy']:.3f}")
    logger.info("="*50)
    
    # Save results
    results_file = Path(output_dir) / "corrected_evaluation_metrics.json"
    with open(results_file, 'w') as f:
        json.dump({
            'overall_metrics': overall_metrics,
            'per_image_metrics': metrics
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function to train and evaluate the corrected model."""
    logger.info("Starting Corrected Building Segmentation Model")
    
    # Set up paths
    images_dir = "test/images"
    labels_dir = "test/labels"
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        logger.error(f"Labels directory not found: {labels_dir}")
        return
    
    # Initialize model
    model = CorrectedBuildingSegmentationModel()
    
    # Check if model already exists
    if model.load_model():
        logger.info("Loaded existing model. Skipping training.")
    else:
        logger.info("No existing model found. Starting training...")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(images_dir, labels_dir, batch_size=2)
        
        # Setup training
        model.setup_training(learning_rate=1e-4)
        
        # Train model
        model.train(train_loader, val_loader, epochs=20)
    
    # Evaluate model
    evaluate_model(model, images_dir, labels_dir, max_images=10)
    
    logger.info("Corrected model evaluation completed!")


if __name__ == "__main__":
    main()
