"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Image Folder Inference Script
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def get_image_files(img_folder, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """获取文件夹中的所有图片文件
    
    Args:
        img_folder: 图片文件夹路径
        extensions: 支持的图片扩展名
    
    Returns:
        image_files: 图片文件路径列表
    """
    image_files = []
    for ext in extensions:
        image_files.extend(Path(img_folder).glob(f'*{ext}'))
        image_files.extend(Path(img_folder).glob(f'*{ext.upper()}'))
    
    # 转换为相对于img_folder的文件名
    image_files = [f.name for f in image_files]
    return sorted(image_files)


def load_category_mapping(config_path):
    """从配置文件加载类别映射
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        category_mapping: 类别ID到类别名的映射字典
    """
    cfg = YAMLConfig(config_path)
    
    # 尝试从配置中获取类别映射
    category_mapping = {}
    
    # 检查是否有train_dataloader配置
    if 'train_dataloader' in cfg.yaml_cfg:
        dataset_cfg = cfg.yaml_cfg['train_dataloader'].get('dataset', {})
        ann_file = dataset_cfg.get('ann_file', '')
        
        if ann_file and os.path.exists(ann_file):
            # 从COCO格式的标注文件加载类别映射
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            for cat in coco_data['categories']:
                category_mapping[cat['id']] = cat['name']
    
    # 如果没有找到类别映射,使用默认的COCO类别
    if not category_mapping:
        print("Warning: No category mapping found in config. Using default class_id format.")
    
    return category_mapping


def process_image_folder(model, device, img_folder, image_files, category_mapping,
                         output_dir, size=(640, 640), conf_threshold=0.01, batch_size=32):
    """处理图片文件夹并生成预测结果
    
    Args:
        model: 推理模型
        device: 设备
        img_folder: 图片文件夹路径
        image_files: 图片文件名列表
        category_mapping: 类别ID到类别名的映射字典
        output_dir: 输出目录
        size: 输入图片尺寸
        conf_threshold: 置信度阈值
        batch_size: 批量推理的批次大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    
    print(f"Processing {len(image_files)} images from {img_folder} with batch_size={batch_size}...")
    
    # 批量处理图片
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        batch_images = []
        batch_orig_sizes = []
        valid_files = []
        
        # 读取并预处理批次中的所有图片
        for img_filename in batch_files:
            img_path = os.path.join(img_folder, img_filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            try:
                im_pil = Image.open(img_path).convert('RGB')
                w, h = im_pil.size
                
                # 图片预处理
                im_data = transforms(im_pil)
                batch_images.append(im_data)
                batch_orig_sizes.append([w, h])
                valid_files.append(img_filename)
            
            except Exception as e:
                print(f"Error loading {img_filename}: {e}")
                # 如果加载出错,保存空结果
                output_filename = os.path.splitext(img_filename)[0] + '.json'
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump([], f)
        
        if not batch_images:
            continue
        
        # 批量推理
        try:
            batch_images_tensor = torch.stack(batch_images).to(device)
            batch_orig_sizes_tensor = torch.tensor(batch_orig_sizes).to(device)
            
            with torch.no_grad():
                output = model(batch_images_tensor, batch_orig_sizes_tensor)
                labels, boxes, scores = output
            
            # 处理每张图片的结果
            for idx, img_filename in enumerate(valid_files):
                # 过滤低置信度检测
                img_labels = labels[idx]
                img_boxes = boxes[idx]
                img_scores = scores[idx]
                
                mask = img_scores > conf_threshold
                img_labels = img_labels[mask]
                img_boxes = img_boxes[mask]
                img_scores = img_scores[mask]
                
                # 构建结果
                results = []
                for label, box, score in zip(img_labels, img_boxes, img_scores):
                    label_id = label.item()
                    # 从类别映射中获取类别名称
                    if category_mapping:
                        category_name = category_mapping.get(label_id, f"class_{label_id}")
                    else:
                        category_name = f"class_{label_id}"
                    
                    result = {
                        "type": category_name,
                        "box2d": [
                            round(box[0].item(), 3),
                            round(box[1].item(), 3),
                            round(box[2].item(), 3),
                            round(box[3].item(), 3)
                        ],
                        "conf": round(score.item(), 2)
                    }
                    results.append(result)
                
                # 保存结果到JSON文件
                output_filename = os.path.splitext(img_filename)[0] + '.json'
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # 如果批次处理出错,为该批次的所有图片保存空结果
            for img_filename in valid_files:
                output_filename = os.path.splitext(img_filename)[0] + '.json'
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump([], f)
    
    print(f"Inference complete. Results saved to: {output_dir}")


def main(args):
    """主函数"""
    # 加载配置
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    # 加载模型权重
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    # 加载模型
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    device = args.device
    model = Model().to(device)
    model.eval()
    
    img_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    
    # 加载类别映射
    category_mapping = load_category_mapping(args.config)
    if category_mapping:
        print(f"Loaded {len(category_mapping)} categories from config")
    
    # 获取图片文件夹中的所有图片
    print(f"Scanning image folder: {args.input}")
    image_files = get_image_files(args.input)
    
    if not image_files:
        print(f"Error: No images found in {args.input}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # 如果指定了输出目录,使用指定的;否则使用默认的
    output_dir = args.output if args.output else 'prediction'
    
    # 执行推理
    process_image_folder(
        model=model,
        device=device,
        img_folder=args.input,
        image_files=image_files,
        category_mapping=category_mapping,
        output_dir=output_dir,
        size=tuple(img_size),
        conf_threshold=args.conf_threshold,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Folder Inference')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input image folder')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for predictions (default: prediction)')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--conf-threshold', type=float, default=0.01,
                        help='Confidence threshold for detections')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    
    args = parser.parse_args()
    main(args)