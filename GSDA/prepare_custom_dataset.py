"""
Prepare custom dataset for training
Converts your video folders into the required JSON format
"""

import os
import json
import random
from pathlib import Path

def prepare_custom_dataset(
    fake_video_dir,
    real_video_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Prepare custom dataset for training
    
    Args:
        fake_video_dir: Path to folder containing fake videos (each video is a subfolder)
        real_video_dir: Path to folder containing real videos (each video is a subfolder)
        output_dir: Where to save the JSON files
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with dataset statistics
    """
    
    random.seed(random_seed)
    
    print("="*70)
    print("Preparing Custom Dataset")
    print("="*70)
    
    # Get all video IDs
    fake_videos = sorted([d for d in os.listdir(fake_video_dir) 
                         if os.path.isdir(os.path.join(fake_video_dir, d))])
    real_videos = sorted([d for d in os.listdir(real_video_dir)
                         if os.path.isdir(os.path.join(real_video_dir, d))])
    
    print(f"\nFound {len(fake_videos)} fake videos")
    print(f"Found {len(real_videos)} real videos")
    
    # Create video list (use video IDs directly, not pairs)
    all_videos = []
    
    # Add fake videos
    for fake_id in fake_videos:
        all_videos.append({
            'id': fake_id,
            'label': 1  # Fake
        })
    
    # Add real videos  
    for real_id in real_videos:
        all_videos.append({
            'id': real_id,
            'label': 0  # Real
        })
    
    # Shuffle
    random.shuffle(all_videos)
    
    # Split
    n_total = len(all_videos)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    train_videos = all_videos[:n_train]
    val_videos = all_videos[n_train:n_train+n_val]
    test_videos = all_videos[n_train+n_val:]
    
    # Convert to simple list of IDs for JSON
    train_ids = [v['id'] for v in train_videos]
    val_ids = [v['id'] for v in val_videos]
    test_ids = [v['id'] for v in test_videos]
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_ids, f, indent=4)
    
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_ids, f, indent=4)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_ids, f, indent=4)
    
    # Statistics
    train_fake = sum(1 for v in train_videos if v['label'] == 1)
    train_real = len(train_videos) - train_fake
    val_fake = sum(1 for v in val_videos if v['label'] == 1)
    val_real = len(val_videos) - val_fake
    test_fake = sum(1 for v in test_videos if v['label'] == 1)
    test_real = len(test_videos) - test_fake
    
    print(f"\n{'='*70}")
    print("Dataset Split Summary")
    print(f"{'='*70}")
    print(f"Train: {len(train_videos)} videos ({train_fake} fake, {train_real} real)")
    print(f"Val:   {len(val_videos)} videos ({val_fake} fake, {val_real} real)")
    print(f"Test:  {len(test_videos)} videos ({test_fake} fake, {test_real} real)")
    print(f"Total: {n_total} videos")
    
    stats = {
        'total_fake': len(fake_videos),
        'total_real': len(real_videos),
        'train_samples': len(train_videos),
        'val_samples': len(val_videos),
        'test_samples': len(test_videos),
        'train_fake': train_fake,
        'train_real': train_real,
        'val_fake': val_fake,
        'val_real': val_real,
        'test_fake': test_fake,
        'test_real': test_real,
        'split_ratio': f"{int(train_ratio*100)}-{int(val_ratio*100)}-{int(test_ratio*100)}",
        'random_seed': random_seed
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n✓ Saved train.json ({len(train_ids)} videos)")
    print(f"✓ Saved val.json ({len(val_ids)} videos)")
    print(f"✓ Saved test.json ({len(test_ids)} videos)")
    print(f"✓ Saved dataset_stats.json")
    print(f"\n{'='*70}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare custom dataset for training")
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to fake videos folder')
    parser.add_argument('--real_dir', type=str, required=True, help='Path to real videos folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSON files')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Prepare dataset
    stats = prepare_custom_dataset(
        fake_video_dir=args.fake_dir,
        real_video_dir=args.real_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Update dataset.py to point to your data folders")
    print(f"2. Run training: python train_effort_c2p.py --data_root {args.output_dir}")
