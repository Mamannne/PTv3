import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import your models and segmentation dataset
from models.PTv3 import PointTransformerV3
from shapenet_part import ShapeNetPartDataset # Assuming this is in datasets/

# =====================================================================
# 1. Classification Dataset (ShapeNetCore) & Collate
# =====================================================================
def read_ply(filename):
    """Parse a PLY file and return vertices as numpy array."""
    vertices = []
    try:
        with open(filename, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            num_vertices = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                    break
            
            for _ in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                if line:
                    parts = line.split()
                    coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                    vertices.append(coords)
        return np.array(vertices, dtype=np.float32)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, cache_path=None, use_cache=True):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.synset_to_label = {} 
        
        self.load_taxonomy() 
        self.load_data()

    def load_taxonomy(self):
        json_path = os.path.join(self.root_dir, 'taxonomy.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                taxonomy = json.load(f)
                for item in taxonomy:
                    self.synset_to_label[item['synsetId']] = item['name'].split(',')[0]

    def load_data(self):
        if self.use_cache and self.cache_path and os.path.isfile(self.cache_path):
            try:
                try:
                    cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
                except TypeError:
                    cache = torch.load(self.cache_path, map_location="cpu")
                self.data = cache["data"]
                self.labels = cache["labels"]
                self.class_to_idx = cache["class_to_idx"]
                print(f"[Dataset] Cache loaded: {len(self.data)} samples")
                return
            except Exception as e:
                print(f"[Dataset] Cache failed ({e}), reloading from disk...")

        synsets = sorted([d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()])
        
        for class_idx, synset_id in enumerate(synsets):
            class_name = self.synset_to_label.get(synset_id, synset_id)
            self.class_to_idx[class_name] = class_idx
            synset_dir = os.path.join(self.root_dir, synset_id)
            all_models = sorted(os.listdir(synset_dir))
            
            model_dirs = [os.path.join(synset_dir, m, 'models', 'model_normalized.ply') 
                          for m in all_models 
                          if os.path.isfile(os.path.join(synset_dir, m, 'models', 'model_normalized.ply'))]

            num_models = len(model_dirs)
            split_idx = int(num_models * 0.85) 
            
            files_to_load = model_dirs[:split_idx] if self.split == 'train' else model_dirs[split_idx:]
            
            for ply_path in files_to_load:
                try:
                    points = read_ply(ply_path)
                    if points is not None and len(points) > 0:
                        self.data.append(points)
                        self.labels.append(class_idx)
                except Exception:
                    pass
            
        if self.cache_path:
            torch.save({"data": self.data, "labels": self.labels, "class_to_idx": self.class_to_idx}, self.cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords = self.data[idx]
        label = self.labels[idx]
        
        if len(coords) >= self.num_points:
            choice = np.random.choice(len(coords), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(coords), self.num_points, replace=True)
        coords = coords[choice]
        
        if self.split == 'train':
            coords = self.translate_pointcloud(coords)
            np.random.shuffle(coords)
        
        coords = torch.from_numpy(coords).float()
        coords = coords - coords.mean(dim=0)
        features = coords.clone()
        coords = coords * 5.0 
        
        return coords, features, label

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        return np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

def packed_collate_fn_cls(batch):
    batch_coords, batch_feats, batch_labels, batch_indices = [], [], [], []
    current_start = 0
    offsets = [0]
    
    for i, (coords, feats, label) in enumerate(batch):
        N = coords.shape[0]
        batch_coords.append(coords)
        batch_feats.append(feats)
        batch_labels.append(label)
        batch_indices.append(torch.full((N,), i, dtype=torch.long))
        current_start += N
        offsets.append(current_start)
        
    return (torch.cat(batch_coords, dim=0), 
            torch.cat(batch_feats, dim=0), 
            torch.tensor(batch_labels, dtype=torch.long),
            torch.cat(batch_indices, dim=0), 
            torch.tensor(offsets, dtype=torch.long))


# =====================================================================
# 2. Segmentation Utils & Collate
# =====================================================================
def packed_collate_fn_seg(batch):
    batch_coords, batch_feats, batch_labels, batch_indices = [], [], [], []
    current_start = 0
    offsets = [0]

    for i, (coords, feats, labels) in enumerate(batch):
        N = coords.shape[0]
        batch_coords.append(coords)
        batch_feats.append(feats)
        batch_labels.append(labels.view(-1))
        batch_indices.append(torch.full((N,), i, dtype=torch.long))
        current_start += N
        offsets.append(current_start)

    return (
        torch.cat(batch_coords, dim=0),
        torch.cat(batch_feats, dim=0),
        torch.cat(batch_labels, dim=0),
        torch.cat(batch_indices, dim=0),
        torch.tensor(offsets, dtype=torch.long),
    )

def compute_class_weights(dataset, num_parts, eps=1e-6):
    counts = np.zeros(num_parts, dtype=np.int64)
    for cat, _p_path, l_path in dataset.datapath:
        labels = np.loadtxt(l_path).astype(np.int64).reshape(-1)
        label_map = dataset.label_maps.get(cat) if dataset.label_maps else None
        if label_map is not None:
            mapped = np.empty_like(labels)
            for raw, mapped_id in label_map.items():
                mapped[labels == raw] = mapped_id
            labels = mapped + int(dataset.label_offsets.get(cat, 0))
        labels = labels[(labels >= 0) & (labels < num_parts)]
        if labels.size:
            counts += np.bincount(labels, minlength=num_parts)

    total = counts.sum()
    if total == 0:
        return torch.ones(num_parts, dtype=torch.float32)

    weights = total / (counts.astype(np.float32) + eps)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


# =====================================================================
# 3. Training Loops
# =====================================================================

def train_segmentation(args, device):
    print(f"\n--- Training Part Segmentation on {device} ---")
    target_classes = ['Airplane', 'Chair', 'Table']
    
    train_data = ShapeNetPartDataset(
        root=args.data_path, split='train', classes=target_classes, npoints=2048,
        augment=True, random_sampling=True, normalize=True, scale=5.0
    )
    test_data = ShapeNetPartDataset(
        root=args.data_path, split='val', classes=target_classes, npoints=2048,
        label_maps=train_data.label_maps, label_offsets=train_data.label_offsets,
        num_parts=train_data.num_parts, augment=False, random_sampling=True, normalize=True, scale=5.0
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=packed_collate_fn_seg, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=packed_collate_fn_seg, num_workers=num_workers, pin_memory=True)
    
    num_parts = train_data.num_parts if train_data.num_parts else 50
    # Assuming PointTransformerV3 takes a backend arg
    model = PointTransformerV3(in_channels=3, num_classes=num_parts, base_grid_size=0.1).to(device)
    
    model_checkpoint = "weights/segmentation_model.pth"
    best_val_acc = 0.0
    if os.path.isfile(model_checkpoint):
        print(f"Loading existing model from {model_checkpoint}...")
        model.load_state_dict(torch.load(model_checkpoint, map_location=device).get("model_state", torch.load(model_checkpoint, map_location=device)))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    class_weights = compute_class_weights(train_data, num_parts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.10 * total_steps))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5)
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_acc, total_points = 0, 0, 0
        
        for i, (coords, feats, labels, b_idx, b_off) in enumerate(train_loader):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            b_idx, b_off = b_idx.to(device), b_off.to(device)
            
            optimizer.zero_grad()
            outputs, order = model(coords, feats, b_idx, b_off, return_order=True)
            labels = labels[order]
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if global_step < warmup_steps:
                warmup_lr = args.lr * float(global_step + 1) / float(warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr
            else:
                scheduler.step()
            global_step += 1
            
            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            total_loss += loss.item()
            total_points += labels.shape[0]
            
            if i % 10 == 0:
                print(f"Ep {epoch} Batch {i}/{len(train_loader)} Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

        print(f"Epoch {epoch} Train Acc: {(total_acc / total_points)*100:.2f}%")
        
        # Eval
        model.eval()
        val_correct, val_points = 0, 0
        with torch.no_grad():
            for coords, feats, labels, b_idx, b_off in test_loader:
                coords, feats, labels, b_idx, b_off = coords.to(device), feats.to(device), labels.to(device), b_idx.to(device), b_off.to(device)
                outputs, order = model(coords, feats, b_idx, b_off, return_order=True)
                labels = labels[order]
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_points += labels.shape[0]
                
        val_acc = (val_correct / val_points) * 100
        print(f"Epoch {epoch} Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "best_val_acc": best_val_acc, "epoch": epoch}, model_checkpoint)
            print(f"New best model saved: {best_val_acc:.2f}%")


def train_classification(args, device):
    print(f"\n--- Starting Classification Training on {device} ---")
    train_dataset = ShapeNetDataset(args.data_path, split='train', num_points=1024, cache_path="data/shapenet_train_cache.pt")
    test_dataset = ShapeNetDataset(args.data_path, split='test', num_points=1024, cache_path="data/shapenet_test_cache.pt")
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=packed_collate_fn_cls, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=packed_collate_fn_cls, num_workers=num_workers, pin_memory=True)
    
    num_classes = len(train_dataset.class_to_idx)
    model = PointTransformerV3(in_channels=3, num_classes=num_classes, base_grid_size=0.1).to(device)
    
    model_checkpoint = "weights/best_ptv3_shapenet_cls.pth"
    best_acc = 0.0
    if os.path.isfile(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        print("Loaded existing classification model.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, batch_count = 0, 0
        
        # Warmup
        if epoch < 5:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * float(epoch + 1) / 5.0

        for batch_i, (coords, feats, labels, b_idx, b_off) in enumerate(train_loader):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            b_idx, b_off = b_idx.to(device), b_off.to(device)
            
            optimizer.zero_grad()
            outputs = model(coords, feats, b_idx, b_off)
            
            # Global Max Pooling for Classification
            batch_size = labels.shape[0]
            fill_value = torch.finfo(outputs.dtype).min
            pooled_logits = outputs.new_full((batch_size, num_classes), fill_value)
            pooled_logits = pooled_logits.scatter_reduce_(0, b_idx.unsqueeze(1).repeat(1, num_classes), outputs, reduce="amax", include_self=False)
            
            loss = criterion(pooled_logits, labels)
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            if (batch_i + 1) % 10 == 0:
                print(f" [Batch {batch_i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        scheduler.step()
        print(f"\n[Epoch {epoch+1}] Average Loss: {total_loss / batch_count:.4f}")
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for coords, feats, labels, b_idx, b_off in test_loader:
                coords, feats, labels, b_idx, b_off = coords.to(device), feats.to(device), labels.to(device), b_idx.to(device), b_off.to(device)
                outputs = model(coords, feats, b_idx, b_off)
                
                pooled_logits = outputs.new_full((labels.shape[0], num_classes), fill_value)
                pooled_logits = pooled_logits.scatter_reduce_(0, b_idx.unsqueeze(1).repeat(1, num_classes), outputs, reduce="amax", include_self=False)
                
                preds = torch.argmax(pooled_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
                
        val_acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), model_checkpoint)
            print(f"🎉 New Best Accuracy: {best_acc:.2f}%")


# =====================================================================
# Main Execution & Argument Parsing
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Transformer V3 Training Script")
    parser.add_argument('--task', type=str, required=True, choices=['segmentation', 'classification'],
                        help="Choose which task to train.")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the dataset directory.")
    parser.add_argument('--backend', type=str, default='hybrid', choices=['naive', 'triton', 'hybrid'],
                        help="Execution backend for the xCPE module.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Training batch size.")
    parser.add_argument('--epochs', type=int, default=80,
                        help="Total number of epochs to train.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Base learning rate.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.task == 'segmentation':
        train_segmentation(args, device)
    elif args.task == 'classification':
        train_classification(args, device)