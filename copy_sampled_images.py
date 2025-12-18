#!/usr/bin/env python3
"""
根据JSON文件复制图片到指定目录
源路径已写死，只需指定目标路径
"""
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# 写死的配置
JSON_FILE = 'instances.json'
IMAGES_DIRS = ['images/train', 'images/val']  # 按优先级查找图片


def copy_single_image(args):
    """复制单张图片"""
    src_path, dst_path = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True, src_path.name
    except Exception as e:
        return False, f"{src_path.name}: {str(e)}"


def find_image(filename, search_dirs):
    """在多个目录中查找图片"""
    for search_dir in search_dirs:
        img_path = Path(search_dir) / filename
        if img_path.exists():
            return img_path
    return None


def copy_images_from_json(dst_dir, num_workers=8):
    """从JSON文件复制所有图片"""

    print(f"\n{'='*70}")
    print(f"从 {JSON_FILE} 复制图片")
    print(f"{'='*70}")

    # 读取JSON文件
    print(f"\n读取JSON: {JSON_FILE}")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    print(f"需要复制 {len(images)} 张图片")

    dst_path = Path(dst_dir)

    # 准备复制任务
    copy_tasks = []
    missing_files = []

    print(f"\n查找图片文件...")
    for img in tqdm(images, desc="查找图片", unit="img"):
        filename = img['file_name']

        # 在多个目录中查找图片
        src_path = find_image(filename, IMAGES_DIRS)

        if src_path:
            dst_file_path = dst_path / filename
            copy_tasks.append((src_path, dst_file_path))
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"\n警告: {len(missing_files)} 个文件未找到")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"  - {f}")

    if not copy_tasks:
        print("没有文件需要复制！")
        return 0

    dst_path.mkdir(parents=True, exist_ok=True)

    # 多线程复制
    print(f"\n目标目录: {dst_path}")
    print(f"开始复制 {len(copy_tasks)} 个文件...\n")

    success_count = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(copy_single_image, task) for task in copy_tasks]

        with tqdm(total=len(copy_tasks), desc="复制进度", unit="file") as pbar:
            for future in as_completed(futures):
                success, info = future.result()
                if success:
                    success_count += 1
                else:
                    failed_files.append(info)
                pbar.update(1)

    print(f"\n成功: {success_count} 个文件")
    if failed_files:
        print(f"失败: {len(failed_files)} 个文件")
        for fail_info in failed_files[:5]:
            print(f"  - {fail_info}")

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='根据JSON文件复制图片到指定目录',
        epilog=f"""
写死的配置:
  JSON文件: {JSON_FILE}
  图片查找目录: {', '.join(IMAGES_DIRS)}

使用示例:
  python copy_sampled_images.py --dst /path/to/output
  python copy_sampled_images.py --dst output_images --workers 16
        """
    )
    parser.add_argument('--dst', type=str, required=True, help='目标目录路径')
    parser.add_argument('--workers', type=int, default=8, help='并行线程数')

    args = parser.parse_args()

    print(f"配置:")
    print(f"  JSON文件: {JSON_FILE}")
    print(f"  图片目录: {IMAGES_DIRS}")
    print(f"  目标目录: {args.dst}")
    print(f"  并行线程: {args.workers}")

    # 执行复制
    total = copy_images_from_json(args.dst, args.workers)

    print(f"\n{'='*70}")
    print(f"全部完成！总共复制: {total} 个文件")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
