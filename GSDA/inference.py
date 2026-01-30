"""
Inference Script for C2P + SVD Detector (BATCH VERSION - NO TEXT GUIDANCE)
支持单张图像和批量文件夹推理，自动处理权重不匹配问题

新增功能：
1. 自动识别输入类型（文件/文件夹）
2. 批量处理多张图像
3. 结果保存为CSV
4. 详细统计信息
5. 宽松权重加载（忽略训练时的文本引导参数）
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from c2p_svd_detector import C2P_SVD_Detector


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with C2P+SVD (Batch Support)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or folder')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for batch results (auto-generated if not specified)')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    parser.add_argument('--svd_rank', type=int, default=1023,
                        help='SVD rank used during training (default: 1023)')
    
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search subdirectories for images')
    
    parser.add_argument('--save_details', action='store_true',
                        help='Save detailed per-image results to separate text files')
    
    return parser.parse_args()


def load_model(checkpoint_path, svd_rank, device):
    """
    加载训练好的模型（宽松加载版本，自动处理权重不匹配）
    
    Args:
        checkpoint_path: 检查点文件路径
        svd_rank: SVD秩（必须与训练时一致）
        device: 设备
    
    Returns:
        model: 加载好的模型
        checkpoint: 检查点字典（包含训练信息）
    """
    print(f"\n🔄 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 打印检查点信息
    print(f"  ✓ Checkpoint loaded")
    print(f"    - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"    - Val AUC: {checkpoint.get('val_auc', 0.0):.4f}")
    print(f"    - Val Acc: {checkpoint.get('val_acc', 0.0):.4f}")
    
    # 初始化推理模型（不使用文本引导）
    print(f"\n🔧 Initializing inference model (SVD rank: {svd_rank})")
    print(f"    - Text guidance: DISABLED (inference mode)")
    
    model = C2P_SVD_Detector(
        clip_model_name='openai/clip-vit-large-patch14',
        num_classes=2,
        svd_rank=svd_rank,
        use_text_guidance=False  # 推理时不需要文本引导
    ).to(device)
    
    # ★ 宽松加载权重 - 自动过滤不匹配的键
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # 过滤：只保留形状匹配的键
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"  ⚠️  Shape mismatch for {k}: checkpoint{v.shape} vs model{model_dict[k].shape}")
    
    # 检测不匹配的键
    unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())
    missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())
    
    # 报告不匹配情况
    if unexpected_keys:
        print(f"\n  ℹ️  Ignoring {len(unexpected_keys)} unexpected keys from checkpoint:")
        for key in sorted(list(unexpected_keys)[:5]):
            print(f"      - {key}")
        if len(unexpected_keys) > 5:
            print(f"      ... and {len(unexpected_keys) - 5} more")
        print(f"  ✓ These keys are from training components not needed for inference")
    
    if missing_keys:
        print(f"\n  ⚠️  Warning: {len(missing_keys)} model parameters not found in checkpoint")
        print(f"      These will use random initialization:")
        for key in sorted(list(missing_keys)[:3]):
            print(f"      - {key}")
        if len(missing_keys) > 3:
            print(f"      ... and {len(missing_keys) - 3} more")
    
    # 加载权重（非严格模式）
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print(f"\n  ✅ Model loaded successfully")
    print(f"      - Loaded parameters: {len(filtered_state_dict)}/{len(model_dict)}")
    print(f"      - Ready for inference\n")
    
    return model, checkpoint


def preprocess_image(image_path):
    """
    预处理单张图像
    
    Args:
        image_path: 图像路径
    
    Returns:
        img_tensor: 预处理后的图像张量 [1, 3, 224, 224]
    """
    # CLIP标准预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
        return img_tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")


def predict(model, image_tensor, device):
    """
    预测单张图像
    
    Args:
        model: 训练好的模型
        image_tensor: 预处理后的图像 [1, 3, 224, 224]
        device: 设备
    
    Returns:
        result_dict: 包含预测结果的字典
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        logits = model(image_tensor)  # [1, 2]
        
        # 计算概率
        probs = F.softmax(logits, dim=1)  # [1, 2]
        
        # 提取结果
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        
        # 预测类别
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = "FAKE" if pred_class == 1 else "REAL"
        
        # 置信度（预测类别的概率）
        confidence = fake_prob if pred_class == 1 else real_prob
    
    result_dict = {
        'prediction': pred_label,
        'confidence': confidence,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'pred_class': pred_class,
        'logits': logits[0].cpu().numpy()
    }
    
    return result_dict


def get_image_files(input_path, recursive=False):
    """
    获取所有图像文件
    
    Args:
        input_path: 输入路径（文件或文件夹）
        recursive: 是否递归搜索子目录
    
    Returns:
        list: 图像文件路径列表
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', 
                       '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF', '.WEBP'}
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix in valid_extensions:
            return [str(input_path)]
        else:
            raise ValueError(f"File is not a valid image: {input_path}")
    
    elif input_path.is_dir():
        # 文件夹
        image_files = []
        
        if recursive:
            # 递归搜索所有子目录
            for ext in valid_extensions:
                image_files.extend(input_path.rglob(f'*{ext}'))
        else:
            # 只搜索当前目录
            for ext in valid_extensions:
                image_files.extend(input_path.glob(f'*{ext}'))
        
        return sorted([str(f) for f in image_files])
    
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def print_single_result(result, image_path):
    """打印单张图像的详细结果"""
    print("\n" + "="*70)
    print("Prediction Results")
    print("="*70)
    print(f"Image:       {os.path.basename(image_path)}")
    print(f"Prediction:  {result['prediction']}")
    print(f"Confidence:  {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Real: {result['real_prob']:.4f} ({result['real_prob']*100:.2f}%)")
    print(f"  Fake: {result['fake_prob']:.4f} ({result['fake_prob']*100:.2f}%)")
    print(f"\nRaw Logits:")
    print(f"  [Real: {result['logits'][0]:.4f}, Fake: {result['logits'][1]:.4f}]")
    
    # 可信度评估
    if result['confidence'] >= 0.9:
        confidence_level = "Very High"
        emoji = "🟢"
    elif result['confidence'] >= 0.75:
        confidence_level = "High"
        emoji = "🟡"
    elif result['confidence'] >= 0.6:
        confidence_level = "Medium"
        emoji = "🟠"
    else:
        confidence_level = "Low"
        emoji = "🔴"
    
    print(f"\nConfidence Level: {emoji} {confidence_level}")
    
    if result['confidence'] < 0.6:
        print("\n⚠️  Warning: Low confidence prediction. The model is uncertain about this image.")
    
    print("="*70 + "\n")


def save_detailed_result(result, image_path, output_dir):
    """保存单张图像的详细结果到文本文件"""
    output_file = os.path.join(output_dir, f"{Path(image_path).stem}_result.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("C2P + SVD Detector - Prediction Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Image:       {os.path.basename(image_path)}\n")
        f.write(f"Full Path:   {image_path}\n")
        f.write(f"Prediction:  {result['prediction']}\n")
        f.write(f"Confidence:  {result['confidence']:.2%}\n\n")
        f.write("Probabilities:\n")
        f.write(f"  Real: {result['real_prob']:.4f} ({result['real_prob']*100:.2f}%)\n")
        f.write(f"  Fake: {result['fake_prob']:.4f} ({result['fake_prob']*100:.2f}%)\n\n")
        f.write("Raw Logits:\n")
        f.write(f"  [Real: {result['logits'][0]:.4f}, Fake: {result['logits'][1]:.4f}]\n")
        f.write("="*70 + "\n")


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("C2P + SVD Detector - Inference Mode (NO TEXT GUIDANCE)")
    print("="*70)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"\n📱 Using device: {args.device}")
    
    # 加载模型
    try:
        model, checkpoint = load_model(args.checkpoint, args.svd_rank, args.device)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取图像文件列表
    print(f"🔍 Scanning input: {args.input}")
    try:
        image_files = get_image_files(args.input, recursive=args.recursive)
    except Exception as e:
        print(f"\n❌ Failed to get image files: {e}")
        return
    
    if not image_files:
        print("❌ No valid images found!")
        return
    
    num_images = len(image_files)
    print(f"  ✓ Found {num_images} image(s)")
    
    # 判断是单图像还是批量
    is_batch = num_images > 1
    
    # 准备输出路径
    if args.output is None:
        if is_batch:
            # 批量：自动生成CSV文件名
            input_name = Path(args.input).name
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"predictions_{input_name}_{timestamp}.csv"
        else:
            # 单图：不需要CSV
            args.output = None
    
    # 创建详细结果输出目录（如果需要）
    detail_output_dir = None
    if args.save_details and is_batch:
        detail_output_dir = f"detailed_results_{Path(args.input).name}"
        os.makedirs(detail_output_dir, exist_ok=True)
        print(f"  ✓ Detailed results will be saved to: {detail_output_dir}")
    
    # 批量推理
    results = []
    errors = []
    
    print(f"\n🚀 Starting inference on {num_images} image(s)...\n")
    
    # 使用进度条（批量模式）或简单处理（单图模式）
    iterator = tqdm(image_files, desc="Processing", unit="img") if is_batch else image_files
    
    for img_path in iterator:
        try:
            # 预处理
            img_tensor = preprocess_image(img_path)
            
            # 预测
            result = predict(model, img_tensor, args.device)
            
            # 记录结果
            results.append({
                'image_path': img_path,
                'filename': os.path.basename(img_path),
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'real_prob': result['real_prob'],
                'fake_prob': result['fake_prob'],
                'logit_real': result['logits'][0],
                'logit_fake': result['logits'][1]
            })
            
            # 单图像模式：直接打印详细结果
            if not is_batch:
                print_single_result(result, img_path)
            
            # 保存详细结果（如果启用）
            if args.save_details and detail_output_dir:
                save_detailed_result(result, img_path, detail_output_dir)
            
        except Exception as e:
            error_msg = f"{os.path.basename(img_path)}: {str(e)}"
            errors.append({'image': img_path, 'error': str(e)})
            if not is_batch:
                print(f"\n❌ Failed to process image: {error_msg}")
            else:
                # 批量模式下更新进度条描述
                if isinstance(iterator, tqdm):
                    iterator.set_postfix_str(f"Error: {os.path.basename(img_path)[:20]}")
    
    # 批量模式：保存CSV并打印统计
    if is_batch and results:
        # 保存CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")
        
        # 统计信息
        print("\n" + "="*70)
        print("📊 Summary Statistics")
        print("="*70)
        print(f"Total images processed: {len(results)}")
        
        num_real = sum(df['prediction'] == 'REAL')
        num_fake = sum(df['prediction'] == 'FAKE')
        
        print(f"Predicted REAL:         {num_real} ({num_real/len(results)*100:.1f}%)")
        print(f"Predicted FAKE:         {num_fake} ({num_fake/len(results)*100:.1f}%)")
        
        print(f"\nConfidence Statistics:")
        print(f"  Mean:    {df['confidence'].mean():.2%}")
        print(f"  Median:  {df['confidence'].median():.2%}")
        print(f"  Std:     {df['confidence'].std():.2%}")
        print(f"  Min:     {df['confidence'].min():.2%}")
        print(f"  Max:     {df['confidence'].max():.2%}")
        
        # 置信度分布
        high_conf = sum(df['confidence'] >= 0.9)
        med_conf = sum((df['confidence'] >= 0.6) & (df['confidence'] < 0.9))
        low_conf = sum(df['confidence'] < 0.6)
        
        print(f"\nConfidence Distribution:")
        print(f"  🟢 Very High (≥90%): {high_conf} ({high_conf/len(results)*100:.1f}%)")
        print(f"  🟡 Medium (60-90%):  {med_conf} ({med_conf/len(results)*100:.1f}%)")
        print(f"  🔴 Low (<60%):       {low_conf} ({low_conf/len(results)*100:.1f}%)")
        
        # 分类统计
        if num_real > 0:
            real_avg_conf = df[df['prediction'] == 'REAL']['confidence'].mean()
            print(f"\nREAL predictions avg confidence: {real_avg_conf:.2%}")
        if num_fake > 0:
            fake_avg_conf = df[df['prediction'] == 'FAKE']['confidence'].mean()
            print(f"FAKE predictions avg confidence: {fake_avg_conf:.2%}")
        
        print("="*70 + "\n")
    
    # 错误报告
    if errors:
        print(f"⚠️  Failed to process {len(errors)} image(s):")
        for err in errors[:10]:  # 只显示前10个错误
            print(f"  - {os.path.basename(err['image'])}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print()
    
    # 最终总结
    if results:
        success_rate = len(results) / (len(results) + len(errors)) * 100
        print(f"✅ Successfully processed {len(results)}/{len(results) + len(errors)} images ({success_rate:.1f}%)\n")


if __name__ == '__main__':
    main()
