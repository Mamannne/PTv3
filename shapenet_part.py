import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

class ShapeNetPartDataset(Dataset):
    def __init__(
        self,
        root='./data/PartAnnotation',
        split='train',
        npoints=2048,
        classes=None,
        label_maps=None,
        label_offsets=None,
        num_parts=None,
        augment=True,
        random_sampling=True,
        normalize=True,
        scale=5.0,
    ):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.cat = {}
        self.label_maps = label_maps
        self.label_offsets = label_offsets
        self.num_parts = num_parts
        self.augment = augment
        self.random_sampling = random_sampling
        self.normalize = normalize
        self.scale = scale
        
        # --- 1. Load Categories ---
        # Try finding the category mapping file in current or parent dir
        catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        if not os.path.exists(catfile):
             catfile = os.path.join(os.path.dirname(self.root), 'synsetoffset2category.txt')
            
        if os.path.exists(catfile):
            with open(catfile, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]
        else:
            print(f"⚠️ Warning: category map not found. Using defaults.")
            self.cat = {'Airplane': '02691156', 'Chair': '03001627', 'Table': '04379243'}

        if classes is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in classes}
        
        print(f"[{split}] Loading classes: {list(self.cat.keys())}")

        # --- 2. Scan Files (Robust Search) ---
        self.datapath = []
        build_label_map = self.label_maps is None or self.label_offsets is None
        if build_label_map:
            self.label_maps = {}
            self.label_offsets = {}
            current_offset = 0
        
        for category, synset_id in self.cat.items():
            # Define potential paths for LABELS
            # Priority 1: Expert Verified
            # Priority 2: Standard Point Labels
            possible_label_dirs = [
                os.path.join(self.root, synset_id, 'expert_verified', 'points_label'),
                os.path.join(self.root, synset_id, 'expert_verified', 'point_labels'),
                os.path.join(self.root, synset_id, 'points_label'),
                os.path.join(self.root, synset_id, 'point_labels'),
                os.path.join(self.root, 'expert_verified', synset_id, 'points_label'),
                os.path.join(self.root, 'expert_verified', synset_id, 'point_labels'),
                os.path.join(self.root, 'points_label', synset_id),
                os.path.join(self.root, 'point_labels', synset_id)
            ]
            
            label_dir = None
            label_files = []
            for p in possible_label_dirs:
                if not os.path.isdir(p):
                    continue
                seg_files = sorted(glob.glob(os.path.join(p, '*.seg')))
                if seg_files:
                    label_dir = p
                    label_files = seg_files
                    break
            
            # Define path for POINTS
            point_dir = os.path.join(self.root, synset_id, 'points')
            if not os.path.isdir(point_dir):
                # Try finding points in a flat structure
                point_dir = os.path.join(self.root, 'points', synset_id)

            if label_dir is None or not os.path.isdir(point_dir):
                print(f"  ❌ Skipping {category}: Could not find folders.")
                print(f"     Checked labels: {possible_label_dirs}")
                continue

            # Load files (only .seg per-point labels)
            if not label_files:
                print(f"  ❌ Skipping {category}: No .seg labels found in {label_dir}.")
                continue

            if build_label_map:
                label_set = set()
                for l_path in label_files:
                    labels = np.loadtxt(l_path).astype(np.int64).reshape(-1)
                    label_set.update(np.unique(labels).tolist())
                label_values = sorted(label_set)
                self.label_maps[category] = {val: idx for idx, val in enumerate(label_values)}
                self.label_offsets[category] = current_offset
                current_offset += len(label_values)
                print(
                    f"  [labels] {category}: {len(label_values)} parts, offset {self.label_offsets[category]}"
                )
            
            # Split (80% Train, 10% Val, 10% Test)
            n = len(label_files)
            if n == 0: continue
            
            if split == 'train':
                selection = label_files[:int(n*0.8)]
            elif split == 'val':
                selection = label_files[int(n*0.8):int(n*0.9)]
            else:
                selection = label_files[int(n*0.9):]
            
            count = 0
            for l_path in selection:
                # Find matching point file
                filename = os.path.basename(l_path)
                model_id = os.path.splitext(filename)[0]
                
                # Check for .pts, .xyz, or .txt
                found_point = False
                for ext in ['.pts', '.xyz', '.txt']:
                    p_path = os.path.join(point_dir, model_id + ext)
                    if os.path.exists(p_path):
                        self.datapath.append((category, p_path, l_path))
                        found_point = True
                        count += 1
                        break
                
            print(f"  ✅ {category}: Loaded {count} samples (Source: {os.path.basename(os.path.dirname(label_dir))})")

        if build_label_map:
            self.num_parts = current_offset
        elif self.num_parts is None:
            max_parts = 0
            for cat, label_map in self.label_maps.items():
                offset = self.label_offsets.get(cat, 0)
                max_parts = max(max_parts, offset + len(label_map))
            self.num_parts = max_parts

    def __getitem__(self, index):
        cat, p_path, l_path = self.datapath[index]
        
        # Load Points & Labels
        # Use try-except to skip corrupted files during training
        try:
            pts = np.loadtxt(p_path).astype(np.float32)
            labels = np.loadtxt(l_path).astype(np.int64).reshape(-1)
        except:
            # Fallback: return random data to avoid crashing (or cleaner: skip index)
            pts = np.zeros((self.npoints, 3), dtype=np.float32)
            labels = np.zeros((self.npoints,), dtype=np.int64)

        # Map to a global part index space when available
        label_map = self.label_maps.get(cat) if self.label_maps else None
        if label_map is not None:
            mapped = np.empty_like(labels)
            for raw, mapped_id in label_map.items():
                mapped[labels == raw] = mapped_id
            labels = mapped + int(self.label_offsets.get(cat, 0))

        # Safety: Ensure lengths match
        if len(pts) != len(labels):
            min_len = min(len(pts), len(labels))
            pts = pts[:min_len]
            labels = labels[:min_len]

        if self.random_sampling:
            if len(pts) >= self.npoints:
                choice = np.random.choice(len(pts), self.npoints, replace=False)
            else:
                choice = np.random.choice(len(pts), self.npoints, replace=True)
        else:
            if len(pts) >= self.npoints:
                choice = np.arange(self.npoints)
            else:
                choice = np.arange(self.npoints) % max(1, len(pts))
            
        pts = pts[choice, :]
        labels = labels[choice]
        
        # Augmentation (Train only)
        if self.split == 'train' and self.augment:
            pts = self.translate_pointcloud(pts)
            indices = np.arange(len(pts))
            np.random.shuffle(indices)
            pts = pts[indices]
            labels = labels[indices]

        if self.normalize:
            pts = pts - np.expand_dims(np.mean(pts, axis=0), 0)
            dist = np.max(np.sqrt(np.sum(pts ** 2, axis=1)), 0)
            if dist > 0:
                pts = pts / dist
        
        if self.scale is not None:
            pts = pts * float(self.scale)
        
        return torch.from_numpy(pts), torch.from_numpy(pts), torch.from_numpy(labels)

    def __len__(self):
        return len(self.datapath)

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        return np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')