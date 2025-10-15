#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版数据格式转换脚本：将自定义格式转换为COCO格式
默认创建训练集、验证集(5000张)和测试集(5000张)
"""

import json
import os
import glob
from PIL import Image
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path
import random


class SimpleDatasetConverter:
    """简化版数据集格式转换器"""
    
    def __init__(self, data_dir: str, label_dir: str, output_dir: str):
        """
        初始化转换器
        
        Args:
            data_dir: 图像数据目录
            label_dir: 标注文件目录  
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO格式数据结构
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 预定义的14个类别（写死）
        self.predefined_categories = [
            {"id": 1, "name": "car", "chinese_name": "小汽车"},
            {"id": 2, "name": "truck", "chinese_name": "货车"},
            {"id": 3, "name": "construction_truck", "chinese_name": "工程车辆"},
            {"id": 4, "name": "van", "chinese_name": "厢式面包车"},
            {"id": 5, "name": "bus", "chinese_name": "巴士"},
            {"id": 6, "name": "bicycle", "chinese_name": "两轮车"},
            {"id": 7, "name": "cyclist", "chinese_name": "两轮车骑行者"},
            {"id": 8, "name": "tricycle", "chinese_name": "三轮车、三轮车骑行者"},
            {"id": 9, "name": "trolley", "chinese_name": "手推车"},
            {"id": 10, "name": "pedestrian", "chinese_name": "行人"},
            {"id": 11, "name": "cone", "chinese_name": "锥形桶、柱形桶"},
            {"id": 12, "name": "barrier", "chinese_name": "水马、栅栏"},
            {"id": 13, "name": "animal", "chinese_name": "小动物"},
            {"id": 14, "name": "other", "chinese_name": "其他"}
        ]
        
        # 初始化类别
        self.coco_data["categories"] = [
            {
                "id": cat["id"],
                "name": cat["name"],
                "supercategory": "object",
                "chinese_name": cat["chinese_name"]
            }
            for cat in self.predefined_categories
        ]
        
        # 类别映射
        self.category_mapping = {cat["name"]: cat["id"] for cat in self.predefined_categories}
        self.annotation_id_counter = 1
        
    def get_image_info(self, image_path: Path, image_id: int) -> Dict[str, Any]:
        """获取图像信息"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
            return {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height
            }
        except Exception as e:
            print(f"警告：无法读取图像 {image_path}: {e}")
            return None
    
    def convert_bbox_format(self, box2d: List[float]) -> List[float]:
        """转换边界框格式：从 [xmin, ymin, xmax, ymax] 到 [x, y, width, height]"""
        xmin, ymin, xmax, ymax = box2d
        width = xmax - xmin
        height = ymax - ymin
        return [xmin, ymin, width, height]
    
    def convert_annotation(self, annotation: Dict[str, Any], image_id: int) -> Dict[str, Any]:
        """转换单个标注"""
        # 获取类别ID
        category_id = self.category_mapping.get(annotation["type"], 14)  # 默认为other
        
        # 转换边界框格式
        bbox = self.convert_bbox_format(annotation["box2d"])
        
        # 计算面积
        area = bbox[2] * bbox[3]  # width * height
        
        # 处理遮挡和截断信息
        occluded = annotation.get("occluded", 0)
        truncated = annotation.get("truncated", 0)
        
        # 根据遮挡程度和截断状态确定iscrowd
        iscrowd = 1 if (occluded >= 3 or truncated == 1) else 0
        
        coco_annotation = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": iscrowd,
            "segmentation": [],
            "occluded": occluded,
            "truncated": truncated
        }
        
        self.annotation_id_counter += 1
        return coco_annotation
    
    def process_single_file(self, image_path: Path, label_path: Path, image_id: int) -> bool:
        """处理单个图像和标注文件"""
        try:
            # 获取图像信息
            image_info = self.get_image_info(image_path, image_id)
            if image_info is None:
                return False
                
            self.coco_data["images"].append(image_info)
            
            # 读取标注文件
            with open(label_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 转换每个标注
            for annotation in annotations:
                coco_annotation = self.convert_annotation(annotation, image_id)
                self.coco_data["annotations"].append(coco_annotation)
            
            return True
            
        except Exception as e:
            print(f"处理文件时出错 {image_path}: {e}")
            return False
    
    def convert_dataset(self) -> None:
        """转换整个数据集"""
        print("开始转换数据集...")
        
        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(ext))
            image_files.extend(self.data_dir.glob(ext.upper()))
        
        image_files.sort()
        
        processed_count = 0
        failed_count = 0
        
        for image_id, image_path in enumerate(image_files, 1):
            # 构造对应的标注文件路径
            label_path = self.label_dir / f"{image_path.stem}.json"
            
            if not label_path.exists():
                print(f"警告：找不到标注文件 {label_path}")
                failed_count += 1
                continue
            
            if self.process_single_file(image_path, label_path, image_id):
                processed_count += 1
            else:
                failed_count += 1
            
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个文件...")
        
        print(f"转换完成！成功处理 {processed_count} 个文件，失败 {failed_count} 个文件")
        print(f"总共发现 {len(self.coco_data['categories'])} 个类别")
        
        # 打印类别信息
        print("类别映射：")
        for category in self.coco_data["categories"]:
            print(f"  {category['id']}: {category['name']}")
    
    def create_splits_and_save(self) -> None:
        """创建数据集分割并保存（写死5000张验证集和5000张测试集）"""
        total_images = len(self.coco_data["images"])
        print(f"总图像数量: {total_images}")
        
        # 固定分割大小
        val_size = 5000
        test_size = 5000
        
        if total_images < val_size + test_size:
            print(f"警告：总图像数量({total_images})小于所需的验证集({val_size})和测试集({test_size})大小")
            # 按比例调整
            ratio = total_images / (val_size + test_size)
            val_size = int(val_size * ratio)
            test_size = int(test_size * ratio)
            print(f"调整后的验证集大小: {val_size}, 测试集大小: {test_size}")
        
        # 创建图像ID列表并随机打乱
        image_ids = list(range(1, total_images + 1))
        random.shuffle(image_ids)
        
        # 分割数据集
        val_image_ids = set(image_ids[:val_size])
        test_image_ids = set(image_ids[val_size:val_size + test_size])
        train_image_ids = set(image_ids[val_size + test_size:])
        
        print(f"训练集: {len(train_image_ids)} 张图像")
        print(f"验证集: {len(val_image_ids)} 张图像")
        print(f"测试集: {len(test_image_ids)} 张图像")
        
        # 创建分割后的数据集
        splits = {
            "train": {"images": [], "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []}
        }
        
        # 分配图像
        for image in self.coco_data["images"]:
            image_id = image["id"]
            if image_id in train_image_ids:
                splits["train"]["images"].append(image)
            elif image_id in val_image_ids:
                splits["val"]["images"].append(image)
            elif image_id in test_image_ids:
                splits["test"]["images"].append(image)
        
        # 分配标注
        for annotation in self.coco_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id in train_image_ids:
                splits["train"]["annotations"].append(annotation)
            elif image_id in val_image_ids:
                splits["val"]["annotations"].append(annotation)
            elif image_id in test_image_ids:
                splits["test"]["annotations"].append(annotation)
        
        # 保存所有分割
        for split in ["train", "val", "test"]:
            split_data = {
                "images": splits[split]["images"],
                "annotations": splits[split]["annotations"],
                "categories": self.coco_data["categories"]
            }
            
            output_file = self.output_dir / f"instances_{split}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            print(f"COCO格式文件已保存到: {output_file}")
            
            # 保存统计信息
            stats = {
                "total_images": len(split_data["images"]),
                "total_annotations": len(split_data["annotations"]),
                "total_categories": len(split_data["categories"]),
                "categories": {cat["name"]: cat["id"] for cat in split_data["categories"]}
            }
            
            stats_file = self.output_dir / f"conversion_stats_{split}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"转换统计信息已保存到: {stats_file}")
        
        print("数据集分割完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将自定义格式数据集转换为COCO格式（自动创建训练/验证/测试集）")
    parser.add_argument("--data_dir", type=str, default="dataset/data", 
                       help="图像数据目录路径")
    parser.add_argument("--label_dir", type=str, default="dataset/label", 
                       help="标注文件目录路径")
    parser.add_argument("--output_dir", type=str, default="dataset/coco_format", 
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.data_dir):
        print(f"错误：图像目录不存在 {args.data_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"错误：标注目录不存在 {args.label_dir}")
        return
    
    # 创建转换器并执行转换
    converter = SimpleDatasetConverter(args.data_dir, args.label_dir, args.output_dir)
    converter.convert_dataset()
    converter.create_splits_and_save()
    
    print("数据集转换完成！已自动创建训练集、验证集(5000张)和测试集(5000张)")


if __name__ == "__main__":
    main()