import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class VectorTokenizer:
    """
    Parses stringified one-hot vectors like "[0. 1. 0. 0. 0.]" into integer class IDs.
    Also handles direct numpy arrays.
    """
    def __init__(self, num_classes=5):
        # +1 for the "unconditional/null" token (index 0)
        self.num_classes = num_classes + 1 

    def encode(self, label_input):
        """
        Input: 
            - String "[0. 1. 0. 0. 0.]"
            - Numpy array [0, 1, 0, 0, 0]
        Output: Integer ID (1-based), e.g., 2
        """
        try:
            # Case 1: Numpy Array (One-hot or Index)
            if isinstance(label_input, np.ndarray):
                # If it's one-hot (e.g. size 5), find argmax
                if label_input.size > 1:
                    return np.argmax(label_input) + 1
                # If it's a single scalar, just return it (assuming 0-indexed in data)
                return int(label_input) + 1
            
            # Case 2: String representation
            if isinstance(label_input, str):
                # Clean string: remove brackets and extra spaces
                clean = label_input.replace('[', '').replace(']', '').strip()
                # Handle space separated values
                parts = [float(x) for x in clean.split()]
                # Find index of max value (usually 1.0)
                idx = parts.index(max(parts))
                return idx + 1
                
            return 0
        except Exception as e:
            return 0 

    def decode(self, token_id):
        if token_id == 0: return "Null"
        return f"Class {token_id}"

class PixelArtDataset(Dataset):
    def __init__(self, root_dir, image_size=16, augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.use_numpy = False
        
        # Define Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 1. Check for Numpy files first
        npy_img_path = os.path.join(root_dir, 'sprites.npy')
        npy_lbl_path = os.path.join(root_dir, 'sprites_labels.npy')

        if os.path.exists(npy_img_path) and os.path.exists(npy_lbl_path):
            print(f"Found numpy files. Loading from {npy_img_path}...")
            self.use_numpy = True
            self.images = np.load(npy_img_path)
            self.labels = np.load(npy_lbl_path)
            self.tokenizer = VectorTokenizer(num_classes=5)
            print(f"Loaded {len(self.images)} images from numpy files.")
            return

        # 2. Fallback to CSV/Folder Logic
        csv_path = os.path.join(root_dir, 'labels.csv')
        
        # Soft search for CSV if exact name doesn't exist
        if not os.path.exists(csv_path):
            files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
            if files:
                csv_path = os.path.join(root_dir, files[0])
            else:
                raise FileNotFoundError(f"No labels.csv or sprites.npy found in {root_dir}"
                                       f"Run 'python setup_data.py' to download the dataset."
                )
        
        print(f"Loading labels from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.tokenizer = VectorTokenizer(num_classes=5)

    def __len__(self):
        if self.use_numpy:
            return len(self.images)
        return len(self.df)

    def __getitem__(self, idx):
        if self.use_numpy:
            # --- Numpy Path ---
            img_arr = self.images[idx]
            
            # Handle Shapes: We want HWC for PIL
            # If shape is (3, 16, 16), transpose to (16, 16, 3)
            if img_arr.shape[0] == 3 and img_arr.shape[2] != 3:
                img_arr = np.transpose(img_arr, (1, 2, 0))
                
            # Handle Dtypes: We want uint8 [0, 255]
            if img_arr.dtype != np.uint8:
                # If float is in [0, 1], scale it
                if img_arr.max() <= 1.0:
                    img_arr = (img_arr * 255).astype(np.uint8)
                else:
                    img_arr = img_arr.astype(np.uint8)
            
            # Convert to PIL for consistent transform pipeline
            image = Image.fromarray(img_arr).convert("RGB")
            
            # Process Label
            label_raw = self.labels[idx]
            label_id = self.tokenizer.encode(label_raw)
            
        else:
            # --- CSV/File Path ---
            row = self.df.iloc[idx]
            rel_path = str(row['Image Path'])
            img_path = os.path.join(self.root_dir, rel_path)
            
            # Handle missing parent directories in path
            if not os.path.exists(img_path):
                 img_path = os.path.join(self.root_dir, 'images', os.path.basename(rel_path))

            label_str = str(row['Label'])
            
            try:
                image = Image.open(img_path).convert("RGB")
                label_id = self.tokenizer.encode(label_str)
            except Exception as e:
                # print(f"Error loading {img_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))

        # Apply transforms (Resize -> Augment -> ToTensor -> Normalize)
        image = self.transform(image)
        
        return image, torch.tensor(label_id).long()

def get_dataloader(root_dir, batch_size=64, image_size=16):
    dataset = PixelArtDataset(root_dir=root_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    return dataloader, dataset.tokenizer