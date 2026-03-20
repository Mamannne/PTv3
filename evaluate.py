import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import models and datasets
from models.PTv3 import PointTransformerV3
from train import ShapeNetDataset, ShapeNetPartDataset, packed_collate_fn_cls, packed_collate_fn_seg
from torch.utils.data import DataLoader

# Standard distinct colors for part segmentation visualization
COLOR_MAP = np.array([
    [255, 0, 0],   [0, 255, 0],   [0, 0, 255],   [255, 255, 0], 
    [255, 0, 255], [0, 255, 255], [128, 0, 0],   [0, 128, 0], 
    [0, 0, 128],   [128, 128, 0], [128, 0, 128], [0, 128, 128]
], dtype=np.float32) / 255.0

def set_axes_equal(ax, coords):
    """Sets 3D plot axes to equal scale."""
    max_range = np.array([coords[:,0].max()-coords[:,0].min(), 
                          coords[:,1].max()-coords[:,1].min(), 
                          coords[:,2].max()-coords[:,2].min()]).max() / 2.0
    mid_x = (coords[:,0].max()+coords[:,0].min()) * 0.5
    mid_y = (coords[:,1].max()+coords[:,1].min()) * 0.5
    mid_z = (coords[:,2].max()+coords[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_axis_off()

def plot_segmentation_pair(coords, labels, preds, title_prefix="Sample"):
    """Plots Ground Truth and Prediction side-by-side."""
    fig = plt.figure(figsize=(14, 7))
    
    # Ground Truth Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=COLOR_MAP[labels % len(COLOR_MAP)], s=20, marker='.')
    ax1.set_title(f"{title_prefix} - Ground Truth", fontweight='bold')
    set_axes_equal(ax1, coords)

    # Prediction Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=COLOR_MAP[preds % len(COLOR_MAP)], s=20, marker='.')
    ax2.set_title(f"{title_prefix} - Prediction", fontweight='bold')
    set_axes_equal(ax2, coords)

    plt.tight_layout()
    plt.show()

def plot_single_point_cloud(coords, labels, title="Point Cloud"):
    """Plots a single 3D point cloud interactively."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=COLOR_MAP[labels % len(COLOR_MAP)], s=20, marker='.')
    ax.set_title(title, fontweight='bold')
    set_axes_equal(ax, coords)
    plt.show()

def unbatch_tensors(coords, labels, preds, b_idx):
    """Separates the packed 1D tensors back into individual point clouds."""
    batch_size = b_idx.max().item() + 1
    unbatched = []
    for i in range(batch_size):
        mask = (b_idx == i)
        unbatched.append((coords[mask], labels[mask], preds[mask]))
    return unbatched

def evaluate_segmentation(args, device):
    print(f"--- Evaluating Part Segmentation on {device} ---")
    target_classes = ['Airplane', 'Chair', 'Table']
    dataset = ShapeNetPartDataset(root=args.data_path, split='val', classes=target_classes, npoints=2048)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=packed_collate_fn_seg)
    
    num_parts = dataset.num_parts if dataset.num_parts else 50
    model = PointTransformerV3(in_channels=3, num_classes=num_parts, base_grid_size=0.1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device).get("model_state", torch.load(args.checkpoint, map_location=device)))
    model.eval()
    
    total_correct, total_points = 0, 0
    plotted_samples = 0
    
    with torch.no_grad():
        for batch_i, (coords, feats, labels, b_idx, b_off) in enumerate(loader):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            b_idx, b_off = b_idx.to(device), b_off.to(device)
            
            outputs, order = model(coords, feats, b_idx, b_off, return_order=True)
            labels = labels[order]
            preds = torch.argmax(outputs, dim=1)
            
            total_correct += (preds == labels).sum().item()
            total_points += labels.shape[0]
            
            if plotted_samples < args.num_visualize:
                coords_ordered = coords[order].cpu().numpy()
                labels_np = labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                b_idx_ordered = b_idx[order].cpu()
                
                unbatched = unbatch_tensors(coords_ordered, labels_np, preds_np, b_idx_ordered)
                
                for pc_coords, pc_labels, pc_preds in unbatched:
                    if plotted_samples >= args.num_visualize:
                        break
                    
                    print(f"Plotting Sample {plotted_samples + 1} / {args.num_visualize} side-by-side...")
                    plot_segmentation_pair(pc_coords, pc_labels, pc_preds, title_prefix=f"Sample {plotted_samples+1}")
                    plotted_samples += 1

    print(f"\n✅ Evaluation Complete!")
    print(f"Overall Accuracy: {(total_correct / total_points) * 100:.2f}%")

def evaluate_classification(args, device):
    print(f"--- Evaluating Classification on {device} ---")
    dataset = dataset = ShapeNetDataset(args.data_path, split='test', num_points=1024, cache_path="/mnt/c/Users/Aymane/OneDrive/shared/NPM3D/proj/data/shapenet_test_cache.pt")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=packed_collate_fn_cls)
    
    num_classes = len(dataset.class_to_idx)
    model = PointTransformerV3(in_channels=3, num_classes=num_classes, base_grid_size=0.1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    correct, total = 0, 0
    worst_conf, best_conf = 1.0, 0.0
    
    with torch.no_grad():
        for batch_i, (coords, feats, labels, b_idx, b_off) in enumerate(loader):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            b_idx, b_off = b_idx.to(device), b_off.to(device)
            
            outputs = model(coords, feats, b_idx, b_off)
            
            batch_size = labels.shape[0]
            fill_value = torch.finfo(outputs.dtype).min
            pooled_logits = outputs.new_full((batch_size, num_classes), fill_value)
            pooled_logits = pooled_logits.scatter_reduce_(0, b_idx.unsqueeze(1).repeat(1, num_classes), outputs, reduce="amax", include_self=False)
            
            probs = torch.softmax(pooled_logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            
            for i in range(batch_size):
                conf = confs[i].item()
                is_correct = (preds[i] == labels[i]).item()
                
                if is_correct and conf > best_conf:
                    best_conf = conf
                    best_pc = coords[b_idx == i].cpu().numpy()
                    best_label = preds[i].item()
                elif not is_correct and conf > worst_conf:
                    worst_conf = conf
                    worst_pc = coords[b_idx == i].cpu().numpy()
                    worst_pred = preds[i].item()
                    worst_true = labels[i].item()

    if 'best_pc' in locals():
        print(f"\nPlotting Best Case: Correctly predicted class {best_label} with {best_conf*100:.1f}% confidence.")
        plot_single_point_cloud(best_pc, np.zeros(len(best_pc), dtype=int), title=f"Best Case (Pred: {best_label})")
        
    if 'worst_pc' in locals():
        print(f"\nPlotting Worst Case: Incorrectly predicted class {worst_pred} (True: {worst_true}) with {worst_conf*100:.1f}% confidence.")
        plot_single_point_cloud(worst_pc, np.zeros(len(worst_pc), dtype=int), title=f"Worst Case (True: {worst_true} | Pred: {worst_pred})")

    print(f"\n✅ Evaluation Complete!")
    print(f"Overall Accuracy: {(correct / total) * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Point Transformer V3")
    parser.add_argument('--task', type=str, required=True, choices=['segmentation', 'classification'])
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained .pth model file")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_visualize', type=int, default=5, help="Number of segmentation samples to plot")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.task == 'segmentation':
        evaluate_segmentation(args, device)
    elif args.task == 'classification':
        evaluate_classification(args, device)