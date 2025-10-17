"""
COCO格式数据集可视化工具
用于可视化COCO格式的目标检测数据集
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_coco_json(json_path):
    """加载COCO格式的JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    return coco_data


def get_category_dict(categories):
    """创建类别ID到类别名称的映射"""
    category_dict = {}
    for cat in categories:
        category_dict[cat['id']] = cat.get('name', f"class_{cat['id']}")
    return category_dict


def generate_colors(num_classes):
    """为每个类别生成随机颜色"""
    np.random.seed(42)
    colors = {}
    for i in range(1, num_classes + 1):
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    return colors


def draw_bbox(image, bbox, label, color, thickness=2):
    """在图像上绘制边界框和标签
    
    Args:
        image: 输入图像
        bbox: 边界框 [x, y, width, height] (COCO格式)
        label: 类别标签
        color: 边界框颜色
        thickness: 线条粗细
    """
    x, y, w, h = [int(v) for v in bbox]
    
    # 绘制矩形框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # 绘制标签背景
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_label = max(y, label_size[1] + 10)
    cv2.rectangle(image, 
                  (x, y_label - label_size[1] - 10),
                  (x + label_size[0], y_label + baseline - 10),
                  color, 
                  cv2.FILLED)
    
    # 绘制标签文本
    cv2.putText(image, label, (x, y_label - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def visualize_coco_dataset(json_path, image_dir, output_dir, num_images=10):
    """可视化COCO格式数据集
    
    Args:
        json_path: COCO格式JSON文件路径
        image_dir: 图像文件夹路径
        output_dir: 可视化结果保存路径
        num_images: 要可视化的图像数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载COCO数据
    print(f"正在加载COCO数据: {json_path}")
    coco_data = load_coco_json(json_path)
    
    # 获取类别信息
    categories = coco_data.get('categories', [])
    category_dict = get_category_dict(categories)
    colors = generate_colors(len(categories))
    
    print(f"数据集包含 {len(categories)} 个类别:")
    for cat_id, cat_name in category_dict.items():
        print(f"  - ID {cat_id}: {cat_name}")
    
    # 构建图像ID到标注的映射
    image_annotations = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 获取所有图像信息
    images = coco_data.get('images', [])
    print(f"\n数据集包含 {len(images)} 张图像")
    
    # 随机选择要可视化的图像
    num_images = min(num_images, len(images))
    selected_images = random.sample(images, num_images)
    
    print(f"\n开始可视化 {num_images} 张图像...")
    
    # 可视化每张图像
    for img_info in tqdm(selected_images, desc="可视化进度"):
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        # 读取图像
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像: {image_path}")
            continue
        
        # 获取该图像的所有标注
        annotations = image_annotations.get(image_id, [])
        
        # 在图像上绘制所有边界框
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_dict.get(category_id, f"class_{category_id}")
            color = colors.get(category_id, (0, 255, 0))
            
            # 构建标签文本
            label = f"{category_name}"
            
            # 绘制边界框
            image = draw_bbox(image, bbox, label, color)
        
        # 在图像顶部添加统计信息
        info_text = f"Image: {file_name} | Objects: {len(annotations)}"
        cv2.putText(image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, f"vis_{file_name}")
        cv2.imwrite(output_path, image)
    
    print(f"\n可视化完成! 结果保存在: {output_dir}")
    print(f"共处理 {num_images} 张图像")


def main():
    """主函数"""
    # 默认参数
    json_path = "dataset/coco_format/instances_train.json"
    image_dir = "dataset/data"
    output_dir = "dataset/vis"
    num_images = 10
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: JSON文件不存在: {json_path}")
        return
    
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return
    
    # 执行可视化
    visualize_coco_dataset(json_path, image_dir, output_dir, num_images)


if __name__ == "__main__":
    main()