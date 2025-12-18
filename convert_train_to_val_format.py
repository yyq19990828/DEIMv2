#!/usr/bin/env python3
"""
将训练集的COCO格式转换为验证集的类别和格式
- 映射类别ID和名称
- 删除验证集中不存在的类别
- 添加supercategory字段
- 修复文件名中的重复前缀问题
- 每N张图片保留一张
"""
import json
from pathlib import Path

# 类别映射：训练集名称 -> 验证集名称
CATEGORY_MAPPING = {
    'car': 'car',
    'truck': 'truck',
    'heavy_truck': 'construction_truck',  # 重命名
    'van': 'van',
    'bus': 'bus',
    'bicycle': 'bicycle',
    'cyclist': 'cyclist',
    'tricycle': 'tricycle',
    'trolley': 'trolley',
    'pedestrian': 'pedestrian',
    'cone': 'cone',
    'animal': 'animal',
    'other': 'other',
    # 'plate': None  # 删除，验证集没有这个类别
}

# 验证集的类别ID映射（从验证集文件中提取）
VAL_CATEGORY_IDS = {
    'car': 1,
    'truck': 2,
    'construction_truck': 3,
    'van': 4,
    'bus': 5,
    'bicycle': 6,
    'cyclist': 7,
    'tricycle': 8,
    'trolley': 9,
    'pedestrian': 10,
    'cone': 11,
    'barrier': 12,  # 训练集没有
    'animal': 13,
    'other': 14,
}


def fix_filename(filename):
    """
    修复文件名中的重复前缀问题
    例如：gpu09-gpu09_xxx.jpg -> gpu09_xxx.jpg
    """
    if '-' in filename:
        parts = filename.split('-', 1)
        if len(parts) == 2:
            prefix = parts[0]
            rest = parts[1]
            # 检查是否是重复前缀模式（如 gpu09-gpu09_xxx）
            if rest.startswith(prefix + '_'):
                return rest
    return filename


def convert_train_to_val_format(train_json_path, output_path, keep_every_n=3):
    """转换训练集到验证集格式，并每N张图片保留一张"""
    print(f"正在读取: {train_json_path}")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    # 构建旧ID到新ID的映射
    old_to_new_cat_id = {}
    valid_old_ids = set()

    for cat in train_data['categories']:
        old_id = cat['id']
        old_name = cat['name']

        # 检查是否需要映射
        if old_name in CATEGORY_MAPPING:
            new_name = CATEGORY_MAPPING[old_name]
            if new_name and new_name in VAL_CATEGORY_IDS:
                new_id = VAL_CATEGORY_IDS[new_name]
                old_to_new_cat_id[old_id] = new_id
                valid_old_ids.add(old_id)
                print(f"映射: {old_name}(ID:{old_id}) -> {new_name}(ID:{new_id})")
            else:
                print(f"删除类别: {old_name}(ID:{old_id}) - 验证集中不存在")

    # 更新categories
    new_categories = []
    for old_name, new_name in CATEGORY_MAPPING.items():
        if new_name and new_name in VAL_CATEGORY_IDS:
            new_categories.append({
                'id': VAL_CATEGORY_IDS[new_name],
                'name': new_name,
                'supercategory': 'object'
            })

    # 按ID排序
    new_categories.sort(key=lambda x: x['id'])

    # 每N张图片保留一张，并修复文件名
    all_images = train_data.get('images', [])
    kept_images = []
    fixed_filename_count = 0

    for idx, img in enumerate(all_images):
        if idx % keep_every_n == 0:
            new_img = img.copy()
            # 修复文件名
            original_filename = new_img['file_name']
            fixed_filename = fix_filename(original_filename)
            if original_filename != fixed_filename:
                fixed_filename_count += 1
            new_img['file_name'] = fixed_filename
            kept_images.append(new_img)

    kept_image_ids = {img['id'] for img in kept_images}

    print(f"\n图片采样 (每{keep_every_n}张保留1张):")
    print(f"原始图片数: {len(all_images)}")
    print(f"保留图片数: {len(kept_images)}")
    print(f"删除图片数: {len(all_images) - len(kept_images)}")
    print(f"修复文件名: {fixed_filename_count}")

    # 过滤和更新annotations（只保留被保留图片的注释）
    new_annotations = []
    removed_cat_count = 0
    removed_img_count = 0

    for ann in train_data.get('annotations', []):
        old_cat_id = ann['category_id']
        image_id = ann['image_id']

        # 检查图片是否被保留
        if image_id not in kept_image_ids:
            removed_img_count += 1
            continue

        # 检查类别是否有效
        if old_cat_id in old_to_new_cat_id:
            ann['category_id'] = old_to_new_cat_id[old_cat_id]
            new_annotations.append(ann)
        else:
            removed_cat_count += 1

    print(f"\n注释统计:")
    print(f"总注释数: {len(train_data.get('annotations', []))}")
    print(f"保留注释数: {len(new_annotations)}")
    print(f"因图片删除而删除: {removed_img_count}")
    print(f"因类别无效而删除: {removed_cat_count}")

    # 构建新的数据结构
    new_data = {
        'images': kept_images,
        'annotations': new_annotations,
        'categories': new_categories,
    }

    # 如果有其他字段，也保留
    for key in train_data:
        if key not in ['images', 'annotations', 'categories']:
            new_data[key] = train_data[key]

    # 保存
    print(f"\n正在保存到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print("转换完成！")
    return new_data


if __name__ == '__main__':
    # 转换 train2017
    print("="*70)
    print("转换 train2017")
    print("="*70)
    convert_train_to_val_format(
        'dataset/coco_format/instances_train2017.json',
        'dataset/coco_format/instances_train2017_converted.json',
        keep_every_n=3
    )

    print("\n" + "="*70)
    print("转换 val2017")
    print("="*70)
    # 转换 val2017
    convert_train_to_val_format(
        'dataset/coco_format/instances_val2017.json',
        'dataset/coco_format/instances_val2017_converted.json',
        keep_every_n=3
    )
