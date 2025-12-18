#!/usr/bin/env python3
"""
高效合并多个COCO格式的JSON文件
自动处理 image_id 和 annotation_id 的冲突
统一类别映射，删除不需要的类别
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

# 写死的源文件列表
SOURCE_FILES = [
    'dataset/coco_format/instances.json',
    'dataset/coco_format/instances_train.json'
]

# 目标类别定义（14类）
TARGET_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "object", "chinese_name": "小汽车"},
    {"id": 2, "name": "truck", "supercategory": "object", "chinese_name": "货车"},
    {"id": 3, "name": "construction_truck", "supercategory": "object", "chinese_name": "工程车辆"},
    {"id": 4, "name": "van", "supercategory": "object", "chinese_name": "厢式面包车"},
    {"id": 5, "name": "bus", "supercategory": "object", "chinese_name": "巴士"},
    {"id": 6, "name": "bicycle", "supercategory": "object", "chinese_name": "两轮车"},
    {"id": 7, "name": "cyclist", "supercategory": "object", "chinese_name": "两轮车骑行者"},
    {"id": 8, "name": "tricycle", "supercategory": "object", "chinese_name": "三轮车、三轮车骑行者"},
    {"id": 9, "name": "trolley", "supercategory": "object", "chinese_name": "手推车"},
    {"id": 10, "name": "pedestrian", "supercategory": "object", "chinese_name": "行人"},
    {"id": 11, "name": "cone", "supercategory": "object", "chinese_name": "锥形桶、柱形桶"},
    {"id": 12, "name": "barrier", "supercategory": "object", "chinese_name": "水马、栅栏"},
    {"id": 13, "name": "animal", "supercategory": "object", "chinese_name": "小动物"},
    {"id": 14, "name": "other", "supercategory": "object", "chinese_name": "其他"}
]

# 类别名称映射（源类别名 -> 目标类别名）
CATEGORY_NAME_MAPPING = {
    'car': 'car',
    'truck': 'truck',
    'heavy_truck': 'construction_truck',  # 合并到 construction_truck
    'construction_truck': 'construction_truck',
    'van': 'van',
    'bus': 'bus',
    'bicycle': 'bicycle',
    'cyclist': 'cyclist',
    'tricycle': 'tricycle',
    'trolley': 'trolley',
    'pedestrian': 'pedestrian',
    'cone': 'cone',
    'barrier': 'barrier',
    'animal': 'animal',
    'other': 'other',
    # 'plate': None  # 删除，不映射
}


def merge_coco_annotations(json_files, output_file):
    """
    合并多个COCO格式的JSON文件

    Args:
        json_files: 源JSON文件路径列表
        output_file: 输出文件路径
    """
    print(f"{'='*70}")
    print("开始合并COCO JSON文件")
    print(f"{'='*70}\n")

    # 建立目标类别名称到ID的映射
    target_name_to_id = {cat['name']: cat['id'] for cat in TARGET_CATEGORIES}

    # 存储合并后的数据
    merged_images = []
    merged_annotations = []

    # ID映射
    new_image_id = 1
    new_ann_id = 1

    total_images = 0
    total_annotations = 0
    removed_annotations = 0

    for idx, json_file in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] 处理: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        # 建立源文件的类别映射
        source_categories = data.get('categories', [])
        old_cat_id_to_name = {cat['id']: cat['name'] for cat in source_categories}

        # 建立旧image_id到新image_id的映射
        old_to_new_image_id = {}

        images = data.get('images', [])
        fixed_filenames = 0
        for img in images:
            old_img_id = img['id']
            old_to_new_image_id[old_img_id] = new_image_id

            # 创建新的image记录
            new_img = img.copy()
            new_img['id'] = new_image_id

            # 清理文件名：修复重复前缀问题（如 gpu09-gpu09 -> gpu09）
            filename = new_img['file_name']
            if '-' in filename:
                parts = filename.split('-', 1)
                if len(parts) == 2:
                    prefix = parts[0]
                    rest = parts[1]
                    # 检查是否是重复前缀模式（如 gpu09-gpu09_xxx）
                    if rest.startswith(prefix + '_'):
                        new_filename = rest
                        new_img['file_name'] = new_filename
                        fixed_filenames += 1

            merged_images.append(new_img)
            new_image_id += 1

        print(f"  图片数: {len(images)} (修复文件名: {fixed_filenames})")

        # 处理annotations
        annotations = data.get('annotations', [])
        file_removed = 0

        for ann in annotations:
            old_img_id = ann['image_id']
            old_cat_id = ann['category_id']

            # 获取源类别名称
            old_cat_name = old_cat_id_to_name.get(old_cat_id)

            # 映射到目标类别名称
            target_cat_name = CATEGORY_NAME_MAPPING.get(old_cat_name)

            # 如果映射为None或不在目标类别中，跳过该注释
            if target_cat_name is None or target_cat_name not in target_name_to_id:
                file_removed += 1
                continue

            # 获取目标类别ID
            target_cat_id = target_name_to_id[target_cat_name]

            # 创建新的annotation记录
            new_ann = ann.copy()
            new_ann['id'] = new_ann_id
            new_ann['image_id'] = old_to_new_image_id[old_img_id]
            new_ann['category_id'] = target_cat_id

            merged_annotations.append(new_ann)
            new_ann_id += 1

        print(f"  注释数: {len(annotations)} (删除: {file_removed})")
        total_images += len(images)
        total_annotations += len(annotations)
        removed_annotations += file_removed

    print(f"\n{'='*70}")
    print("合并统计:")
    print(f"{'='*70}")
    print(f"总图片数: {total_images}")
    print(f"总注释数: {total_annotations}")
    print(f"保留注释: {len(merged_annotations)}")
    print(f"删除注释: {removed_annotations}")
    print(f"总类别数: {len(TARGET_CATEGORIES)}")

    # 统计每个类别的注释数
    cat_counts = Counter(ann['category_id'] for ann in merged_annotations)
    cat_id_to_info = {cat['id']: cat for cat in TARGET_CATEGORIES}

    print(f"\n类别分布:")
    for cat in TARGET_CATEGORIES:
        cat_id = cat['id']
        count = cat_counts.get(cat_id, 0)
        print(f"  ID {cat_id:2d} ({cat['name']:20s}): {count:8,} 个注释")

    # 构建最终数据
    merged_data = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': TARGET_CATEGORIES
    }

    # 保存合并后的文件
    print(f"\n正在保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("合并完成！")
    print(f"{'='*70}\n")

    return merged_data


def verify_merged_file(json_file):
    """验证合并后的文件"""
    print(f"{'='*70}")
    print("验证合并结果")
    print(f"{'='*70}\n")

    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # 检查类别数量
    assert len(categories) == 14, f"类别数量错误: {len(categories)}, 期望: 14"
    print(f"✓ 类别数量正确: 14")

    # 检查image_id的唯一性
    image_ids = [img['id'] for img in images]
    assert len(image_ids) == len(set(image_ids)), "Image IDs 有重复！"
    print(f"✓ Image IDs 唯一性检查通过")

    # 检查annotation_id的唯一性
    ann_ids = [ann['id'] for ann in annotations]
    assert len(ann_ids) == len(set(ann_ids)), "Annotation IDs 有重复！"
    print(f"✓ Annotation IDs 唯一性检查通过")

    # 检查所有annotation的image_id都存在
    image_id_set = set(image_ids)
    for ann in annotations:
        assert ann['image_id'] in image_id_set, f"Annotation {ann['id']} 引用了不存在的 image_id {ann['image_id']}"
    print(f"✓ Annotation image_id 引用检查通过")

    # 检查所有annotation的category_id都存在
    category_id_set = {cat['id'] for cat in categories}
    for ann in annotations:
        assert ann['category_id'] in category_id_set, f"Annotation {ann['id']} 引用了不存在的 category_id {ann['category_id']}"
    print(f"✓ Annotation category_id 引用检查通过")

    # 检查没有 heavy_truck 和 plate
    cat_names = {cat['name'] for cat in categories}
    assert 'heavy_truck' not in cat_names, "不应该包含 heavy_truck"
    assert 'plate' not in cat_names, "不应该包含 plate"
    assert 'barrier' in cat_names, "应该包含 barrier"
    assert 'truck' in cat_names, "应该包含 truck"
    print(f"✓ 类别名称检查通过（已删除 heavy_truck 和 plate）")

    print(f"\n{'='*70}")
    print("验证通过！文件格式正确")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    output_file = 'dataset/coco_format/instances_merged.json'

    print(f"\n源文件:")
    for f in SOURCE_FILES:
        print(f"  - {f}")
    print(f"\n输出文件: {output_file}\n")

    # 合并文件
    merge_coco_annotations(SOURCE_FILES, output_file)

    # 验证结果
    verify_merged_file(output_file)
