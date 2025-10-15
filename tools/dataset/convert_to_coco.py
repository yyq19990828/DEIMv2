#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换脚本：将自定义格式转换为COCO格式
支持图像标注结构说明中的数据格式转换
"""

import json
import os
import glob
from PIL import Image
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path


class DatasetConverter:
    """数据集格式转换器"""
    
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
        
        # 预定义的类别映射（根据标注类别表）
        self.predefined_categories = {
            "car": {"id": 1, "name": "car", "chinese_name": "小汽车",
                   "description": "双轴小型车辆，包括7座及以下的轿车、SUV、皮卡、出租车、无人驾驶车辆、网约车"},
            "truck": {"id": 2, "name": "truck", "chinese_name": "货车",
                     "description": "大中型货车，包括厢式货车、栅式货车、半挂车、平板运输货车、牵引车头、油罐车、混凝土搅拌车、消防车、环卫洒水车、泥头车、渣土车等"},
            "construction_truck": {"id": 3, "name": "construction_truck", "chinese_name": "工程车辆",
                                  "description": "包括叉车、铲车、起重机、压路机、挖土机、推土机、吊车、矿车等工程车辆"},
            "van": {"id": 4, "name": "van", "chinese_name": "厢式面包车",
                   "description": "中型客运或货运面包车、救护车等"},
            "bus": {"id": 5, "name": "bus", "chinese_name": "巴士",
                   "description": "大型客运车辆，包括校巴、旅游巴、客运班车、城市公交等"},
            "bicycle": {"id": 6, "name": "bicycle", "chinese_name": "两轮车",
                       "description": "没有人骑着的两轮车，包含两轮摩托车、两轮电动车、两轮自行车"},
            "cyclist": {"id": 7, "name": "cyclist", "chinese_name": "两轮车骑行者",
                       "description": "有人骑着的两轮车，包含两轮摩托车、两轮电动车、两轮自行车"},
            "tricycle": {"id": 8, "name": "tricycle", "chinese_name": "三轮车、三轮车骑行者",
                        "description": "包含三轮摩托车、三轮电动车、三轮自行车等，包含有人或者无人的情况，若有人骑着需要把人也框住"},
            "trolley": {"id": 9, "name": "trolley", "chinese_name": "手推车",
                       "description": "包含手推车、婴儿车、轮椅。手推车不包含推车的人。婴儿车和轮椅如果有人坐着，应该标为一个整体"},
            "pedestrian": {"id": 10, "name": "pedestrian", "chinese_name": "行人",
                          "description": "包含各种姿态的行人，站立、蹲坐、躺着、静止与动态等，包含人的所有部位，手持任何其他东西不框进"},
            "cone": {"id": 11, "name": "cone", "chinese_name": "锥形桶、柱形桶",
                    "description": "施工、交通意外中用于隔离的锥形桶、柱形防撞桶等"},
            "barrier": {"id": 12, "name": "barrier", "chinese_name": "水马、栅栏",
                       "description": "施工、交通意外中用于隔离的水马、栅栏等"},
            "animal": {"id": 13, "name": "animal", "chinese_name": "小动物",
                      "description": "包含猫、狗等小动物"},
            "other": {"id": 14, "name": "other", "chinese_name": "其他",
                     "description": "车道上影响车辆正常驾驶的其他物体，例如路面碎片或落在车道上的树木等。不包括路上不影响驾驶的小物体，比如一小块垃圾、石头等"}
        }
        
        # 初始化预定义类别到COCO categories
        self._initialize_categories()
        
        # 类别映射和计数器
        self.category_mapping = {cat_info["name"]: cat_info["id"] for cat_info in self.predefined_categories.values()}
        self.category_id_counter = 15  # 从15开始，为未知类别预留
        self.annotation_id_counter = 1
    
    def _initialize_categories(self) -> None:
        """初始化预定义类别到COCO格式"""
        for cat_info in self.predefined_categories.values():
            self.coco_data["categories"].append({
                "id": cat_info["id"],
                "name": cat_info["name"],
                "supercategory": "object",
                "chinese_name": cat_info["chinese_name"],
                "description": cat_info["description"]
            })
        
    def get_image_info(self, image_path: Path, image_id: int) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image_path: 图像文件路径
            image_id: 图像ID
            
        Returns:
            图像信息字典
        """
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
        """
        转换边界框格式：从 [xmin, ymin, xmax, ymax] 到 [x, y, width, height]
        
        Args:
            box2d: 原始边界框坐标 [xmin, ymin, xmax, ymax]
            
        Returns:
            COCO格式边界框 [x, y, width, height]
        """
        xmin, ymin, xmax, ymax = box2d
        width = xmax - xmin
        height = ymax - ymin
        return [xmin, ymin, width, height]
    
    def get_or_create_category_id(self, category_name: str) -> int:
        """
        获取或创建类别ID
        
        Args:
            category_name: 类别名称
            
        Returns:
            类别ID
        """
        # 如果是预定义类别，直接返回对应ID
        if category_name in self.category_mapping:
            return self.category_mapping[category_name]
        
        # 如果是未知类别，创建新的类别
        print(f"警告：发现未知类别 '{category_name}'，将分配新的ID {self.category_id_counter}")
        
        self.category_mapping[category_name] = self.category_id_counter
        
        # 添加到categories列表
        self.coco_data["categories"].append({
            "id": self.category_id_counter,
            "name": category_name,
            "supercategory": "unknown",
            "chinese_name": f"未知类别_{category_name}",
            "description": f"数据集中发现的未预定义类别: {category_name}"
        })
        
        self.category_id_counter += 1
        return self.category_mapping[category_name]
    
    def convert_annotation(self, annotation: Dict[str, Any], image_id: int) -> Dict[str, Any]:
        """
        转换单个标注
        
        Args:
            annotation: 原始标注数据
            image_id: 图像ID
            
        Returns:
            COCO格式标注
        """
        # 获取类别ID
        category_id = self.get_or_create_category_id(annotation["type"])
        
        # 转换边界框格式
        bbox = self.convert_bbox_format(annotation["box2d"])
        
        # 计算面积
        area = bbox[2] * bbox[3]  # width * height
        
        # 根据图像标注结构说明处理遮挡和截断信息
        # occluded: 0表示0%~30%遮挡，1表示30%~50%遮挡，2表示50%~70%遮挡，3表示超过70%遮挡
        # truncated: 0表示不截断，1表示截断
        occluded = annotation.get("occluded", 0)
        truncated = annotation.get("truncated", 0)
        
        # 根据遮挡程度和截断状态确定iscrowd
        # 如果遮挡程度很高(>70%)或者被截断，可以考虑标记为crowd
        iscrowd = 1 if (occluded >= 3 or truncated == 1) else 0
        
        coco_annotation = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": iscrowd,
            "segmentation": [],  # 如果没有分割信息，设为空列表
            # 保留原始的遮挡和截断信息作为额外属性
            "occluded": occluded,
            "truncated": truncated
        }
        
        self.annotation_id_counter += 1
        return coco_annotation
    
    def process_single_file(self, image_path: Path, label_path: Path, image_id: int) -> bool:
        """
        处理单个图像和标注文件
        
        Args:
            image_path: 图像文件路径
            label_path: 标注文件路径
            image_id: 图像ID
            
        Returns:
            是否处理成功
        """
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
    
    def save_coco_format(self, split_name: str = "train") -> None:
        """
        保存COCO格式文件
        
        Args:
            split_name: 数据集分割名称（train/val/test）
        """
        output_file = self.output_dir / f"instances_{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"COCO格式文件已保存到: {output_file}")
        
        # 保存统计信息
        stats = {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "total_categories": len(self.coco_data["categories"]),
            "categories": {cat["name"]: cat["id"] for cat in self.coco_data["categories"]}
        }
        
        stats_file = self.output_dir / f"conversion_stats_{split_name}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"转换统计信息已保存到: {stats_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将自定义格式数据集转换为COCO格式")
    parser.add_argument("--data_dir", type=str, default="dataset/data", 
                       help="图像数据目录路径")
    parser.add_argument("--label_dir", type=str, default="dataset/label", 
                       help="标注文件目录路径")
    parser.add_argument("--output_dir", type=str, default="dataset/coco_format", 
                       help="输出目录路径")
    parser.add_argument("--split_name", type=str, default="train", 
                       help="数据集分割名称 (train/val/test)")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.data_dir):
        print(f"错误：图像目录不存在 {args.data_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"错误：标注目录不存在 {args.label_dir}")
        return
    
    # 创建转换器并执行转换
    converter = DatasetConverter(args.data_dir, args.label_dir, args.output_dir)
    converter.convert_dataset()
    converter.save_coco_format(args.split_name)
    
    print("数据集转换完成！")


if __name__ == "__main__":
    main()