"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
COCO Dataset Inference Script
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




def load_coco_dataset_info(config_path, split='val'):
    """从配置文件加载COCO数据集信息
    
    Args:
        config_path: 配置文件路径
        split: 'val' 或 'test'
    
    Returns:
        img_folder: 图片文件夹路径
        ann_file: 标注文件路径
        category_mapping: 类别ID到类别名的映射
    """
    cfg = YAMLConfig(config_path)
    
    # 根据split选择对应的dataloader配置
    if split == 'val':
        dataloader_cfg = cfg.yaml_cfg.get('val_dataloader', {})
    elif split == 'test':
        # 如果有test_dataloader配置则使用,否则使用val_dataloader
        dataloader_cfg = cfg.yaml_cfg.get('test_dataloader', cfg.yaml_cfg.get('val_dataloader', {}))
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split}")
    
    dataset_cfg = dataloader_cfg.get('dataset', {})
    img_folder = dataset_cfg.get('img_folder', '')
    ann_file = dataset_cfg.get('ann_file', '')
    
    if not img_folder or not ann_file:
        raise ValueError(f"Cannot find img_folder or ann_file in config for split={split}")
    
    # 加载COCO标注文件获取类别映射
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建类别映射: category_id -> category_name
    category_mapping = {}
    for cat in coco_data['categories']:
        category_mapping[cat['id']] = cat['name']
    
    # 构建图片ID到文件名的映射
    image_id_to_filename = {}
    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']
    
    return img_folder, ann_file, category_mapping, image_id_to_filename


def process_coco_dataset(model, device, img_folder, image_files, category_mapping, output_dir, size=(640, 640), conf_threshold=0.01):
    """处理COCO数据集并生成预测结果
    
    Args:
        model: 推理模型
        device: 设备
        img_folder: 图片文件夹路径
        image_files: 图片文件名列表
        category_mapping: 类别ID到类别名的映射字典
        output_dir: 输出目录
        size: 输入图片尺寸
        conf_threshold: 置信度阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    
    print(f"Processing {len(image_files)} images...")
    
    for img_filename in tqdm(image_files, desc="Inference"):
        # 读取图片
        img_path = os.path.join(img_folder, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        try:
            im_pil = Image.open(img_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)
            
            # 图片预处理
            im_data = transforms(im_pil).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                output = model(im_data, orig_size)
                labels, boxes, scores = output
            
            # 过滤低置信度检测
            labels = labels[0]
            boxes = boxes[0]
            scores = scores[0]
            
            mask = scores > conf_threshold
            labels = labels[mask]
            boxes = boxes[mask]
            scores = scores[mask]
            
            # 构建结果
            results = []
            for label, box, score in zip(labels, boxes, scores):
                label_id = label.item()
                # 从标注文件的类别映射中获取类别名称
                category_name = category_mapping.get(label_id, f"class_{label_id}")
                
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
            # 使用图片文件名(不含扩展名)作为输出文件名
            output_filename = os.path.splitext(img_filename)[0] + '.json'
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            # 如果出错,保存空结果
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
    
    # 加载数据集信息
    print(f"Loading COCO dataset info from config: {args.config}")
    print(f"Split: {args.split}")
    
    img_folder, ann_file, category_mapping, image_id_to_filename = load_coco_dataset_info(
        args.config, args.split
    )
    
    print(f"Image folder: {img_folder}")
    print(f"Annotation file: {ann_file}")
    print(f"Number of categories: {len(category_mapping)}")
    print(f"Number of images: {len(image_id_to_filename)}")
    
    # 获取所有图片文件名
    image_files = list(image_id_to_filename.values())
    
    # 如果指定了输出目录,使用指定的;否则使用默认的
    output_dir = args.output if args.output else f'prediction_{args.split}'
    
    # 执行推理
    process_coco_dataset(
        model=model,
        device=device,
        img_folder=img_folder,
        image_files=image_files,
        category_mapping=category_mapping,
        output_dir=output_dir,
        size=tuple(img_size),
        conf_threshold=args.conf_threshold
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO Dataset Inference')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to use (val or test)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for predictions (default: prediction_{split})')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--conf-threshold', type=float, default=0.01,
                        help='Confidence threshold for detections')
    
    args = parser.parse_args()
    main(args)