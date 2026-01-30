# GSDA
# Quick Start - Custom Dataset Training

## 3-Step Process

### **Step 1: Organize Data** (5 minutes)

```bash
# Your data structure should look like:
your_dataset/
├── fake_picture/
│   ├── fake_001/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   └── ...
│   └── fake_002/
└── real_picture/
    ├── real_001/
    └── real_002/
```

### **Step 2: Generate Splits** (1 minute)

```bash
python prepare_custom_dataset.py \
    --fake_dir your_dataset/fake_videos \
    --real_dir your_dataset/real_videos \
    --output_dir your_dataset
```

**Output**: Creates `train.json`, `val.json`, `test.json`

### **Step 3: Modify & Train** (2 minutes setup + training time)

**Option A: Quick modify existing training script**

Edit `train_effort_c2p.py`, replace dataloader section:

```python
# OLD (line ~280):
from dataset import DeepfakeDataset

# NEW:
from dataset_custom import get_custom_dataloader

# OLD dataloader creation:
train_dataset = DeepfakeDataset(...)

# NEW:
train_loader = get_custom_dataloader(
    data_root=args.data_root,
    json_file=os.path.join(args.data_root, 'train.json'),
    batch_size=config['training']['train_batch_size'],
    num_workers=config['training']['workers'],
    num_frames=config['data']['frame_num']['train'],
    fake_folder='fake_picture',  # Your folder name
    real_folder='real_picture',  # Your folder name
    is_train=True
)

val_loader = get_custom_dataloader(
    data_root=args.data_root,
    json_file=os.path.join(args.data_root, 'val.json'),
    batch_size=config['training']['test_batch_size'],
    num_workers=config['training']['workers'],
    num_frames=config['data']['frame_num']['test'],
    fake_folder='fake_picture',
    real_folder='real_picture',
    is_train=False
)
```

**Then run**:
```bash
python train_effort_c2p.py \
    --config effort_c2p_config.yaml \
    --data_root your_dataset
```

---

**Option B: Use custom config**

Create `custom_config.yaml`:

```yaml
# Just copy effort_c2p_config.yaml and change data_root
data_root: "/path/to/your_dataset"

# Everything else stays the same
```

Then:
```bash
python train_effort_c2p.py --config custom_config.yaml
```

---

## What to Expect

### **Small Dataset** :
```
Epoch 1: Val AP ~0.95

```

---

## One-Line Setup

If you have videos in folders:

```bash
# Generate splits and train
python prepare_custom_dataset.py \
    --fake_dir data/fake \
    --real_dir data/real \
    --output_dir data && \
python train_effort_c2p.py \
    --data_root data \
    --config effort_c2p_config.yaml
```

---

## Common Issues

### **Issue**: "PICTURE not found"
**Fix**: Check folder names match in dataset loader:
```python
fake_folder='PICTURE'  # Must match your folder name
real_folder='PICTURE'  # Must match your folder name
```

### **Issue**: CUDA OOM
**Fix**: Reduce batch size in config:
```yaml
train_batch_size: 8  # Instead of 16
```



---

## Files Created

1. `prepare_custom_dataset.py` - Generate splits
2. `dataset_custom.py` - Custom dataloader  
3. `CUSTOM_DATASET_GUIDE.md` - Full guide

**Just modify `train_effort_c2p.py` to use `dataset_custom.py` and you're done!**

---

## Summary

```bash
# 1. Organize: fake_PICTURE/ and real_PICTURE/
# 2. Generate splits
python prepare_custom_dataset.py --fake_dir ... --real_dir ... --output_dir ...

# 3. Modify train_effort_c2p.py to import dataset_custom

# 4. Train
python train_effort_c2p.py --data_root your_dataset
```

**That's it!**
