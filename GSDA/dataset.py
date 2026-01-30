"""
Dataset for Deepfake Detection
支持两种模式：
1. JSON标注文件模式（推荐）
2. 文件夹结构模式（兼容旧代码）

文件夹结构：
root_dir/
  ├── train/
  │   ├── folder_01/
  │   │   ├── 0_real/
  │   │   │   └── *.jpg
  │   │   └── 1_fake/
  │   │       └── *.jpg
  │   ├── folder_02/
  │   └── ...
  └── val/
      └── (同train结构)
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """
    深度伪造检测数据集
    支持JSON标注和文件夹结构两种模式
    """
    
    def __init__(
        self, 
        data_root,
        json_path=None,      # 新增：JSON标注文件路径
        split='train',       # 如果不用JSON，指定split
        transform=None,
        caption_root=None,
        mode='json'          # 'json' 或 'folder'
    ):
        """
        Args:
            data_root: 数据根目录 (例如 D:/svd-c2p/xunlian)
            json_path: JSON标注文件路径 (推荐模式)
            split: 'train' 或 'val' (仅在folder模式下使用)
            transform: 图像变换
            caption_root: Caption文本根目录 (可选)
            mode: 'json' (从JSON加载) 或 'folder' (扫描文件夹)
        """
        self.data_root = Path(data_root)
        self.caption_root = Path(caption_root) if caption_root else None
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        
        # 根据模式构建数据集
        if mode == 'json' and json_path:
            self._load_from_json(json_path)
        elif mode == 'folder':
            self._load_from_folder(split)
        else:
            raise ValueError(f"Invalid mode '{mode}' or missing json_path")
        
        print(f"✓ Loaded {len(self.samples)} samples")
        self._print_stats()
    
    def _load_from_json(self, json_path):
        """
        从JSON标注文件加载数据
        
        JSON格式:
        [
            {
                "image_path": "train/folder_01/0_real/img.jpg",
                "label": 0,
                "label_name": "real"
            },
            ...
        ]
        """
        print(f"\nLoading dataset from JSON: {json_path}")
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        for item in data_list:
            # 图像路径（相对于data_root）
            img_relative_path = item['image_path']
            img_full_path = self.data_root / img_relative_path
            
            # Caption路径（如果有）
            caption_path = None
            if self.caption_root:
                caption_relative = Path(img_relative_path).with_suffix('.txt')
                caption_path = self.caption_root / caption_relative
            
            self.samples.append({
                'image_path': str(img_full_path),
                'label': item['label'],  # 0 or 1
                'label_name': item.get('label_name', 'unknown'),
                'caption_path': str(caption_path) if caption_path else None
            })
    
    def _load_from_folder(self, split):
        """
        从文件夹结构加载数据
        
        结构：root_dir/split/folderX/0_real or 1_fake/*.jpg
        """
        print(f"\nScanning folder structure: {self.data_root / split}")
        
        split_dir = self.data_root / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # 遍历所有子文件夹
        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            # 检查 0_real 和 1_fake
            for class_folder in ['0_real', '1_fake']:
                class_path = folder / class_folder
                
                if not class_path.exists():
                    continue
                
                label = 0 if class_folder == '0_real' else 1
                label_name = 'real' if label == 0 else 'fake'
                
                # 遍历图像文件
                for img_file in sorted(class_path.glob('*')):
                    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        continue
                    
                    # Caption路径
                    caption_path = None
                    if self.caption_root:
                        relative_path = img_file.relative_to(self.data_root)
                        caption_path = self.caption_root / relative_path.with_suffix('.txt')
                    
                    self.samples.append({
                        'image_path': str(img_file),
                        'label': label,
                        'label_name': label_name,
                        'caption_path': str(caption_path) if caption_path else None
                    })
    
    def _print_stats(self):
        """打印数据集统计信息"""
        num_real = sum(1 for s in self.samples if s['label'] == 0)
        num_fake = len(self.samples) - num_real
        
        print(f"  - Real: {num_real}")
        print(f"  - Fake: {num_fake}")
        
        # ★ 修复除零错误
        total = num_real + num_fake
        if total > 0:
            print(f"  - Balance: {num_real/total*100:.1f}% real")
        else:
            print(f"  - Balance: N/A (empty dataset)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: [3, 224, 224], 转换后的图像张量
            label: int, 0 (real) or 1 (fake)
            text: str, 文本描述（label + caption）
        """
        sample = self.samples[idx]
        
        # ========== 1. 加载图像 ==========
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠ Error loading {sample['image_path']}: {e}")
            # 返回黑色图像作为fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # ========== 2. 获取标签 ==========
        label = sample['label']
        
        # ========== 3. 构建文本描述 ==========
        # 基础label文本
        label_text = "a photo of a real face" if label == 0 else "a photo of a fake face"
        
        # 加载caption（如果存在）
        caption_text = ""
        if sample['caption_path'] and os.path.exists(sample['caption_path']):
            try:
                with open(sample['caption_path'], 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            except Exception as e:
                caption_text = ""
        
        # 组合最终文本
        if caption_text:
            full_text = f"{label_text}. {caption_text}"
        else:
            full_text = label_text
        
        return image, label, full_text


def get_transforms(mode='train', resolution=224):
    """
    获取数据变换
    
    Args:
        mode: 'train' 或 'test'
        resolution: 目标分辨率
    
    Returns:
        torchvision.transforms.Compose
    """
    # CLIP官方归一化参数
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # test/val
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def create_dataloaders(
    data_root,
    train_json=None,
    val_json=None,
    caption_root=None,
    batch_size=32,
    num_workers=4,
    mode='json'
):
    """
    创建训练和验证数据加载器
    
    Args:
        data_root: 数据根目录
        train_json: 训练集JSON路径 (json模式必需)
        val_json: 验证集JSON路径 (json模式必需)
        caption_root: Caption根目录 (可选)
        batch_size: 批量大小
        num_workers: 数据加载线程数
        mode: 'json' 或 'folder'
    
    Returns:
        train_loader, val_loader
    """
    
    print("\n" + "="*70)
    print("Creating DataLoaders")
    print("="*70)
    
    # ========== 训练集 ==========
    if mode == 'json':
        if not train_json:
            raise ValueError("train_json is required in JSON mode")
        
        train_dataset = DeepfakeDataset(
            data_root=data_root,
            json_path=train_json,
            transform=get_transforms(mode='train'),
            caption_root=caption_root,
            mode='json'
        )
    else:
        train_dataset = DeepfakeDataset(
            data_root=data_root,
            split='train',
            transform=get_transforms(mode='train'),
            caption_root=caption_root,
            mode='folder'
        )
    
    # ========== 验证集 ==========
    if mode == 'json':
        if not val_json:
            raise ValueError("val_json is required in JSON mode")
        
        val_dataset = DeepfakeDataset(
            data_root=data_root,
            json_path=val_json,
            transform=get_transforms(mode='test'),
            caption_root=caption_root,
            mode='json'
        )
    else:
        val_dataset = DeepfakeDataset(
            data_root=data_root,
            split='val',
            transform=get_transforms(mode='test'),
            caption_root=caption_root,
            mode='folder'
        )
    
    # ========== 创建DataLoader ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 训练时丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing DeepfakeDataset")
    print("="*70)
    
    # ========== 测试参数 ==========
    data_root = r"D:\svd-c2p\xunlian"
    train_json = r"D:\svd-c2p\xunlian\train.json"
    val_json = r"D:\svd-c2p\xunlian\val.json"
    
    # ========== 测试JSON模式 ==========
    print("\n🔵 Test 1: JSON mode")
    print("-" * 70)
    
    try:
        train_loader, val_loader = create_dataloaders(
            data_root=data_root,
            train_json=train_json,
            val_json=val_json,
            batch_size=4,
            num_workers=0,  # 测试时用0避免多进程问题
            mode='json'
        )
        
        # 测试加载一个batch
        print("\nLoading a batch from train_loader...")
        for images, labels, texts in train_loader:
            print(f"  ✓ Images shape: {images.shape}")
            print(f"  ✓ Labels: {labels.tolist()}")
            print(f"  ✓ Text samples:")
            for i, text in enumerate(texts[:2]):
                print(f"    [{i}] {text}")
            break
        
        print("\n✅ JSON mode test passed!")
        
    except Exception as e:
        print(f"\n❌ JSON mode test failed: {e}")
    
    # ========== 测试Folder模式 ==========
    print("\n🟢 Test 2: Folder mode")
    print("-" * 70)
    
    try:
        folder_root = r"D:\svd-c2p\xunlian"
        
        train_loader, val_loader = create_dataloaders(
            data_root=folder_root,
            batch_size=4,
            num_workers=0,
            mode='folder'
        )
        
        print("\nLoading a batch from train_loader...")
        for images, labels, texts in train_loader:
            print(f"  ✓ Images shape: {images.shape}")
            print(f"  ✓ Labels: {labels.tolist()}")
            break
        
        print("\n✅ Folder mode test passed!")
        
    except Exception as e:
        print(f"\n❌ Folder mode test failed: {e}")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")
