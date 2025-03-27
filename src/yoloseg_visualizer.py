import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

class YOLOSegmentationVisualizer:
    def __init__(self, model_path="yolov8s-seg.pt"):
        self.model = YOLO(model_path)
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Key layers to monitor for segmentation visualization
        target_layers = {
            'early': 2,         # Early features (edges, basic shapes)
            'mid': 4,           # Mid-level features
            'deep': 8,          # Deep features (semantic)
            'neck': 12,         # Feature pyramid
            'seg_head': -1      # Final segmentation head output
        }
        
        # Register forward hooks for each target layer
        for name, idx in target_layers.items():
            layer = self.model.model.model[idx]

            # Create hook function to capture activations
            def get_hook_fn(layer_name):
                def hook_fn(module, input, output):
                    # Handle tuple or tensor output
                    if isinstance(output, tuple):
                        # Take the first tensor if output is a tuple
                        output = output[0]
                    
                    # Ensure we're working with a tensor and can call detach()
                    if hasattr(output, 'detach'):
                        self.activations[layer_name] = output.detach()
                    else:
                        # Fallback if detach is not possible
                        self.activations[layer_name] = output
                
                return hook_fn

            # Register the hook
            hook = layer.register_forward_hook(get_hook_fn(name))
            self.hooks.append(hook)

        # Handle segmentation head separately
        # Find the segmentation-specific modules in the model
        # This is more robust as it doesn't rely on a specific attribute name
        for name, module in self.model.model.named_modules():
            if 'segment' in name.lower() or 'seg' in name.lower():
                # Create hook function for segmentation components
                def get_seg_hook_fn(layer_name):
                    def hook_fn(module, input, output):
                        # Handle tuple or tensor output
                        if isinstance(output, tuple):
                            # Take the first tensor if output is a tuple
                            output = output[0]
                        
                        # Ensure we're working with a tensor and can call detach()
                        if hasattr(output, 'detach'):
                            self.activations['seg_head'] = output.detach()
                        else:
                            # Fallback if detach is not possible
                            self.activations['seg_head'] = output
                    return hook_fn

                # Register the hook
                hook = module.register_forward_hook(get_seg_hook_fn('seg_head'))
                self.hooks.append(hook)
                break


    def process_image(self, img_path, conf_threshold=0.25):
        """Process image and return both detections and segmentation masks"""
        # Run inference with segmentation model
        results = self.model(img_path, conf=conf_threshold)
        
        # Load original image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path.copy()
            
        return img, results[0]
    
    def visualize_segmentation_activations(self, img_path, layer_name='deep', class_name=None):
        """Generate heatmap showing activations for segmentation features"""
        # Process image
        original_img, results = self.process_image(img_path)
        
        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in activations")
        
        # Get activations for the requested layer
        activations = self.activations[layer_name]
        
        # Get segmentation masks
        if not hasattr(results, 'masks') or results.masks is None:
            raise ValueError("No segmentation masks found in results")
        
        # Find mask for the specified class
        target_mask = None
        mask_idx = None
        
        if class_name:
            for i, (box, cls) in enumerate(zip(results.boxes.data, results.boxes.cls)):
                if results.names[int(cls)] == class_name:
                    mask_idx = i
                    break
        else:
            # Use the first mask if no class specified
            mask_idx = 0
            
        if mask_idx is None:
            if class_name:
                raise ValueError(f"No mask found for class {class_name}")
            else:
                raise ValueError("No masks found in results")
        
        # Get the target mask
        target_mask = results.masks.data[mask_idx].cpu().numpy()
        
        # Resize mask to match activation size for correlation calculation
        act_size = activations.shape[-2:]
        resized_mask = cv2.resize(target_mask, (act_size[1], act_size[0]))
        
        # Find channels with highest correlation to target mask
        correlations = []
        for c in range(activations.shape[1]):
            channel_act = activations[0, c].cpu().numpy()
            correlation = np.corrcoef(channel_act.flatten(), resized_mask.flatten())[0, 1]
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0
            correlations.append(correlation)
        
        # Get top correlated channels
        top_k = 16  # Number of top channels to use
        top_channels = np.argsort(correlations)[-top_k:]
        
        # Create activation map from top channels
        activation_map = activations[0, top_channels].sum(dim=0).cpu().numpy()
        
        # Normalize activation map
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-10)
        
        # Resize activation map to original image size
        heatmap = cv2.resize(activation_map, (original_img.shape[1], original_img.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        alpha = 0.6
        overlay = original_img.copy()
        overlay = (1-alpha) * original_img + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Create segmentation boundary visualization
        seg_vis = original_img.copy()
        
        # Resize mask to match original image
        orig_mask = cv2.resize(target_mask, (original_img.shape[1], original_img.shape[0]))
        
        # Create contour overlay for segmentation boundaries
        contours, _ = cv2.findContours(
            (orig_mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        cv2.drawContours(seg_vis, contours, -1, (0, 255, 0), 2)
        
        # Combine with heatmap inside the segmentation region
        mask_rgb = np.stack([orig_mask]*3, axis=2)
        seg_heatmap = (1-alpha) * original_img + alpha * heatmap_colored * mask_rgb
        seg_heatmap = np.clip(seg_heatmap, 0, 255).astype(np.uint8)
        
        return original_img, overlay, seg_vis, seg_heatmap
    
    def visualize_all_layers_segmentation(self, img_path, class_name=None):
        """Generate heatmaps for all monitored layers with segmentation focus"""
        # Process image
        original_img, results = self.process_image(img_path)
        
        # If class_name is provided, find relevant mask
        target_cls = None
        if class_name:
            for i, cls in enumerate(results.boxes.cls):
                if results.names[int(cls)] == class_name:
                    target_cls = int(cls)
                    break
        
        # Prepare visualization grid - original + masks + layers
        n_layers = len(self.activations)
        fig, axes = plt.subplots(2, n_layers+1, figsize=(5*(n_layers+1), 10))
        
        # Plot original image with segmentation
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        
        # Apply all segmentation masks
        seg_overlay = original_img.copy()
        
        if hasattr(results, 'masks') and results.masks is not None:
            # Create a colormap for distinct mask colors
            colors = plt.cm.rainbow(np.linspace(0, 1, len(results.boxes.cls)))
            
            for i, (mask, cls, conf) in enumerate(zip(results.masks.data, results.boxes.cls, results.boxes.conf)):
                # Skip if not the target class
                if target_cls is not None and int(cls) != target_cls:
                    continue
                    
                # Get class name and color
                class_name = results.names[int(cls)]
                color = colors[i][:3] * 255  # RGB
                
                # Create mask overlay
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (original_img.shape[1], original_img.shape[0]))
                
                # Apply mask with color
                colored_mask = np.zeros_like(original_img, dtype=np.float32)
                for c in range(3):
                    colored_mask[:,:,c] = mask_np * color[c]
                
                # Blend original with colored mask
                alpha = 0.5
                seg_overlay = np.where(
                    np.expand_dims(mask_np, axis=2) > 0.5,
                    seg_overlay * (1-alpha) + colored_mask * alpha,
                    seg_overlay
                )
                
                # Draw contours
                contours, _ = cv2.findContours(
                    (mask_np * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(seg_overlay, contours, -1, color, 2)
        
        # Show segmentation overlay
        axes[0, 1].imshow(seg_overlay.astype(np.uint8))
        axes[0, 1].set_title("Segmentation Masks")
        
        # Plot each layer's activation heatmap
        layer_idx = 2
        for layer_name in self.activations.keys():
            if layer_idx >= axes.shape[1]:
                continue
                
            # Get activation for this layer
            activations = self.activations[layer_name]
            
            # Create heatmap (sum across channels)
            activation_map = activations.sum(dim=1).squeeze().cpu().numpy()
            
            # Normalize
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-10)
            
            # Resize to original image size
            heatmap = cv2.resize(activation_map, (original_img.shape[1], original_img.shape[0]))
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            alpha = 0.7
            overlay = (1-alpha) * original_img + alpha * heatmap_colored
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Show in grid
            axes[0, layer_idx].imshow(overlay)
            axes[0, layer_idx].set_title(f"Layer: {layer_name}")
            layer_idx += 1
            
        # Second row: Segmentation-specific activations
        if hasattr(results, 'masks') and results.masks is not None and len(results.masks) > 0:
            # Show first mask 
            mask_idx = 0
            if target_cls is not None:
                for i, cls in enumerate(results.boxes.cls):
                    if int(cls) == target_cls:
                        mask_idx = i
                        break
            
            # Show original mask
            mask = results.masks.data[mask_idx].cpu().numpy()
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            
            # Create binary mask visualization
            mask_vis = np.zeros_like(original_img)
            mask_vis[mask > 0.5] = [0, 255, 0]  # Green for mask
            
            # Show mask
            axes[1, 0].imshow(mask_vis)
            axes[1, 0].set_title(f"Mask: {results.names[int(results.boxes.cls[mask_idx])]}")
            
            # Show mask boundaries
            boundary_vis = original_img.copy()
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(boundary_vis, contours, -1, (0, 255, 0), 2)
            axes[1, 1].imshow(boundary_vis)
            axes[1, 1].set_title("Segmentation Boundary")
            
            # Show per-layer activations weighted by mask correlation
            layer_idx = 2
            for layer_name in self.activations.keys():
                if layer_idx >= axes.shape[1]:
                    continue
                    
                # Get activations
                activations = self.activations[layer_name]
                
                # Resize mask to match activation size
                act_size = activations.shape[-2:]
                resized_mask = cv2.resize(mask, (act_size[1], act_size[0]))
                
                # Calculate correlation between each channel and the mask
                correlations = []
                for c in range(activations.shape[1]):
                    channel_act = activations[0, c].cpu().numpy()
                    corr = np.corrcoef(channel_act.flatten(), resized_mask.flatten())[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    correlations.append(corr)
                
                # Get correlation-weighted activation map
                correlations = torch.tensor(correlations).to(activations.device)
                weighted_activations = activations[0] * correlations.view(-1, 1, 1)
                weighted_map = weighted_activations.sum(dim=0).cpu().numpy()
                
                # Normalize
                weighted_map = (weighted_map - weighted_map.min()) / (weighted_map.max() - weighted_map.min() + 1e-10)
                
                # Resize to original image size
                heatmap = cv2.resize(weighted_map, (original_img.shape[1], original_img.shape[0]))
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # Create mask-constrained overlay
                mask_3ch = np.stack([mask]*3, axis=2)
                overlay = original_img.copy()
                # Apply heatmap only within mask
                overlay = np.where(
                    mask_3ch > 0.5,
                    (1-alpha) * original_img + alpha * heatmap_colored,
                    original_img
                )
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                
                # Show in grid
                axes[1, layer_idx].imshow(overlay)
                axes[1, layer_idx].set_title(f"{layer_name} (mask correlation)")
                layer_idx += 1
        
        # Remove axis ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout()
        return fig


    @staticmethod
    def calculate_correlation(channel_act, resized_mask):
        """
        Calculate correlation between channel activation and mask with size normalization.
        
        Args:
        channel_act (numpy.ndarray): Channel activation array
        resized_mask (numpy.ndarray): Resized mask array
        
        Returns:
        float: Correlation coefficient
        """
        # Ensure both inputs are flattened
        channel_act_flat = channel_act.flatten()
        mask_flat = resized_mask.flatten()
        
        # If sizes are different, normalize by sampling or truncating
        if len(channel_act_flat) != len(mask_flat):
            # Choose the smaller length to avoid memory issues
            min_length = min(len(channel_act_flat), len(mask_flat))
            
            # Randomly sample or truncate to match lengths
            if len(channel_act_flat) > min_length:
                indices = np.random.choice(len(channel_act_flat), min_length, replace=False)
                channel_act_flat = channel_act_flat[indices]
            elif len(mask_flat) > min_length:
                indices = np.random.choice(len(mask_flat), min_length, replace=False)
                mask_flat = mask_flat[indices]
        
        # Calculate correlation, handling potential NaN
        try:
            corr = np.corrcoef(channel_act_flat, mask_flat)[0, 1]
            return corr if not np.isnan(corr) else 0
        except Exception:
            # Fallback to zero correlation if calculation fails
            return 0


    def explain_segmentation_decision(self, img_path, class_name=None):
        """Generate detailed explanation of segmentation decision process"""
        # Process image
        original_img, results = self.process_image(img_path)
        
        # Find target class index if provided
        target_idx = None
        if class_name:
            for i, cls in enumerate(results.boxes.cls):
                if results.names[int(cls)] == class_name:
                    target_idx = i
                    break
        else:
            # Use first detection by default
            if len(results.boxes) > 0:
                target_idx = 0
                
        if target_idx is None:
            if class_name:
                raise ValueError(f"No detection found for class {class_name}")
            else:
                raise ValueError("No detections found")
        
        # Get target class and confidence
        target_cls = int(results.boxes.cls[target_idx])
        target_class_name = results.names[target_cls]
        confidence = results.boxes.conf[target_idx].item()
        
        # Get mask for target
        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[target_idx].cpu().numpy()
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
        else:
            raise ValueError("No segmentation mask found")
            
        # Create visualization grid - more detailed for explaining decision
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        
        # Show segmentation mask
        mask_overlay = original_img.copy()
        mask_3ch = np.stack([mask]*3, axis=2)
        alpha = 0.6
        color_mask = np.zeros_like(original_img)
        color_mask[:,:] = [0, 255, 0]  # Green for mask
        mask_overlay = np.where(
            mask_3ch > 0.5,
            mask_overlay * (1-alpha) + color_mask * alpha,
            mask_overlay
        )
        
        # Add contour
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_overlay, contours, -1, (0, 255, 0), 2)
        
        axes[0, 1].imshow(mask_overlay.astype(np.uint8))
        axes[0, 1].set_title(f"{target_class_name} Segmentation (Conf: {confidence:.2f})")
        
        # Show boundary confidence
        # Calculate distance transform for boundary visualization
        dist_transform = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Invert for boundary emphasis
        boundary_conf = 1.0 - dist_transform
        # Apply threshold to focus on boundary
        boundary_conf[boundary_conf < 0.7] = 0
        
        # Create boundary confidence visualization
        boundary_vis = original_img.copy()
        boundary_heatmap = cv2.applyColorMap(np.uint8(boundary_conf * 255), cv2.COLORMAP_HOT)
        boundary_heatmap = cv2.cvtColor(boundary_heatmap, cv2.COLOR_BGR2RGB)
        
        boundary_vis = np.where(
            np.expand_dims(boundary_conf, axis=2) > 0,
            boundary_vis * 0.3 + boundary_heatmap * 0.7,
            boundary_vis
        )
        
        axes[0, 2].imshow(boundary_vis.astype(np.uint8))
        axes[0, 2].set_title("Boundary Confidence")
        
        # Pick two informative layers to show: deep and seg_head
        informative_layers = ['deep', 'seg_head']
        for i, layer_name in enumerate(informative_layers):
            if layer_name not in self.activations:
                continue
                
            # Get activations for this layer
            activations = self.activations[layer_name]
            
            # Resize mask to match activation resolution
            act_size = activations.shape[-2:]
            resized_mask = cv2.resize(mask, (act_size[1], act_size[0]))
            
            # Calculate channel correlations with mask
            correlations = []
            for c in range(activations.shape[1]):
                channel_act = activations[0, c].cpu().numpy()
                #corr = np.corrcoef(channel_act.flatten(), resized_mask.flatten())[0, 1]
                corr = self.calculate_correlation(channel_act, resized_mask)
                if np.isnan(corr):
                    corr = 0
                correlations.append((c, corr))
            
            # Sort by correlation strength
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 3 most correlated channels
            top_channels = [c for c, _ in correlations[:3]]
            
            # Show top 3 channels as RGB composite
            rgb_composite = np.zeros((original_img.shape[0], original_img.shape[1], 3))
            for j, channel in enumerate(top_channels[:3]):
                # Get channel activation
                channel_act = activations[0, channel].cpu().numpy()
                # Normalize
                channel_act = (channel_act - channel_act.min()) / (channel_act.max() - channel_act.min() + 1e-10)
                # Resize
                channel_resized = cv2.resize(channel_act, (original_img.shape[1], original_img.shape[0]))
                # Add to composite
                if j < 3:
                    rgb_composite[:, :, j] = channel_resized
            
            # Normalize composite
            rgb_composite = np.clip(rgb_composite, 0, 1)
            
            # Create overlay with original
            composite_overlay = original_img * 0.5 + rgb_composite * 255 * 0.5
            composite_overlay = np.clip(composite_overlay, 0, 255).astype(np.uint8)
            
            axes[1, i].imshow(composite_overlay)
            axes[1, i].set_title(f"{layer_name} Key Channels")
            
            # Add correlation values as text
            text_info = []
            for j, (channel, corr) in enumerate(correlations[:3]):
                text_info.append(f"Ch {channel}: corr={corr:.2f}")
                
            axes[1, i].text(
                10, 20, 
                "\n".join(text_info),
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=9,
                color='black'
            )
        
        # Show overall feature importance
        if 'deep' in self.activations:
            # Use deep features for overall importance map
            deep_acts = self.activations['deep']
            
            # Sum across channels, weighted by correlation with mask
            act_size = deep_acts.shape[-2:]
            resized_mask = cv2.resize(mask, (act_size[1], act_size[0]))
            
            # Calculate correlation-weighted importance
            importance_map = np.zeros((act_size[0], act_size[1]))
            
            for c in range(deep_acts.shape[1]):
                channel_act = deep_acts[0, c].cpu().numpy()
                #corr = np.corrcoef(channel_act.flatten(), resized_mask.flatten())[0, 1]
                corr = self.calculate_correlation(channel_act, resized_mask)
                if np.isnan(corr):
                    corr = 0
                # Add weighted contribution
                importance_map += channel_act * abs(corr)
            
            # Normalize
            importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-10)
            
            # Resize to original image size
            importance_map = cv2.resize(importance_map, (original_img.shape[1], original_img.shape[0]))
            
            # Apply colormap
            importance_colored = cv2.applyColorMap(np.uint8(importance_map * 255), cv2.COLORMAP_JET)
            importance_colored = cv2.cvtColor(importance_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            alpha = 0.7
            importance_overlay = original_img * (1-alpha) + importance_colored * alpha
            importance_overlay = np.clip(importance_overlay, 0, 255).astype(np.uint8)
            
            axes[1, 2].imshow(importance_overlay)
            axes[1, 2].set_title("Feature Importance Map")
        
        # Remove axis ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout()
        return fig

"""# Example usage
#file_name = "/home/krishna/Krishna/Projects/Python/xAI/easy-explain/easy_explain-main/tests/traffic-light_2.jpeg"
file_name = "/home/krishna/Krishna/Projects/Python/xAI/easy-explain/easy_explain-main/tests/traffic-light_100316101_l.jpg"

class_name= "traffic light"
visualizer = YOLOSegmentationVisualizer("yolov8s-seg.pt")
fig = visualizer.visualize_all_layers_segmentation(file_name, class_name=class_name)
plt.show()

# For detailed explanation of a specific segmentation
fig_explanation = visualizer.explain_segmentation_decision(file_name, class_name=class_name)
plt.show()
"""

import os
import argparse
import matplotlib.pyplot as plt

def get_next_iteration_folder(base_name, base_output_dir='outputs'):
    """
    Generate the next iteration folder name by finding the next available number.
    
    Args:
        base_name (str): Base name for folder naming
        base_output_dir (str, optional): Base directory for output. Defaults to 'outputs'.
    
    Returns:
        str: Full path to the new folder
        int: Iteration number
    """
    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Clean base name (remove special characters)
    clean_base_name = ''.join(c if c.isalnum() else '_' for c in base_name)
    
    # Full path for this specific output group
    group_output_dir = os.path.join(base_output_dir, clean_base_name)
    
    # Ensure group output directory exists
    os.makedirs(group_output_dir, exist_ok=True)
    
    # Find existing folders with this base name
    existing_folders = [
        f for f in os.listdir(group_output_dir)
        if os.path.isdir(os.path.join(group_output_dir, f))
        and f.startswith('iteration_')
    ]
    
    # Determine the next iteration number
    if not existing_folders:
        iteration = 1
    else:
        # Extract numbers from existing folder names and find the max
        iterations = [
            int(folder.split('_')[-1])
            for folder in existing_folders
            if folder.split('_')[-1].isdigit()
        ]
        iteration = max(iterations) + 1 if iterations else 1
    
    # Format the folder name
    folder_name = f"iteration_{iteration:02d}"
    full_path = os.path.join(group_output_dir, folder_name)
    
    # Create the folder
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, iteration

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='YOLO Segmentation Visualization Tool')
    parser.add_argument('--model', type=str, default='yolov8s-seg.pt', 
                        help='Path to YOLO segmentation model')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to input image file')
    parser.add_argument('--target_class', type=str, default=None, 
                        help='Specific class to visualize segmentation for')
    parser.add_argument('--mode', type=str, choices=['all', 'explain'], default='all',
                        help='Visualization mode: all layers or detailed explanation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = YOLOSegmentationVisualizer(args.model)
    
    # Prepare output folder
    # Extract image filename without extension
    image_base = os.path.splitext(os.path.basename(args.image))[0]
    
    # Generate unique folder name based on image and model
    model_base = os.path.splitext(os.path.basename(args.model))[0]
    output_base = f"{image_base}_{model_base}"
    output_folder, iteration = get_next_iteration_folder(output_base)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Run visualization based on mode
    if args.mode == 'all':
        fig = visualizer.visualize_all_layers_segmentation(args.image, class_name=args.target_class)
        plt.suptitle(f"All Layer Segmentation: {os.path.basename(args.image)}")
    else:
        fig = visualizer.explain_segmentation_decision(args.image, class_name=args.target_class)
        plt.suptitle(f"Segmentation Explanation: {os.path.basename(args.image)}")
    
    # Save plot
    plot_path = os.path.join(output_folder, 'segmentation_visualization.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free up memory
    
    # Optional: Save original image to output folder
    import shutil
    shutil.copy(args.image, os.path.join(output_folder, os.path.basename(args.image)))
    
    print(f"Results saved to: {output_folder}")
    print(f"Iteration: {iteration}")

# Add this at the end of the script
if __name__ == '__main__':
    main()

"""
# Visualize all layers for a specific class
python script.py --image /path/to/image.jpg --class "traffic light"

# Use a specific model
python script.py --model custom_model.pt --image /path/to/image.jpg

# Get detailed explanation
python script.py --image /path/to/image.jpg --mode explain
"""