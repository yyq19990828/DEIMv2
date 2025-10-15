#!/bin/bash

# 简化版数据集转换脚本
# 自动创建训练集、验证集(5000张)和测试集(5000张)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DATA_DIR="dataset/data"
LABEL_DIR="dataset/label"
OUTPUT_DIR="dataset/coco_format"

# 显示帮助信息
show_help() {
    echo -e "${BLUE}数据集转换脚本 - 简化版${NC}"
    echo -e "${BLUE}自动创建训练集、验证集(5000张)和测试集(5000张)${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --data-dir DIR      图像数据目录 (默认: dataset/data)"
    echo "  -l, --label-dir DIR     标注文件目录 (默认: dataset/label)"
    echo "  -o, --output-dir DIR    输出目录 (默认: dataset/coco_format)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 -d my_data -l my_labels           # 指定数据和标注目录"
    echo "  $0 -o output/coco                    # 指定输出目录"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -l|--label-dir)
            LABEL_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            echo "使用 '$0 --help' 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证参数
echo -e "${BLUE}=== 数据集转换脚本 - 简化版 ===${NC}"
echo -e "${YELLOW}参数检查:${NC}"
echo "  图像目录: $DATA_DIR"
echo "  标注目录: $LABEL_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 图像目录不存在: $DATA_DIR${NC}"
    exit 1
fi

if [ ! -d "$LABEL_DIR" ]; then
    echo -e "${RED}错误: 标注目录不存在: $LABEL_DIR${NC}"
    exit 1
fi

# 检查Python脚本是否存在
SCRIPT_PATH="tools/dataset/convert_to_coco_simple.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}错误: 转换脚本不存在: $SCRIPT_PATH${NC}"
    exit 1
fi

# 执行转换
echo -e "${GREEN}开始转换数据集...${NC}"
echo ""

python3 "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --label_dir "$LABEL_DIR" \
    --output_dir "$OUTPUT_DIR"

# 检查转换结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 转换完成! ===${NC}"
    echo -e "${YELLOW}输出文件:${NC}"
    
    # 列出生成的文件
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  COCO格式文件:"
        ls -la "$OUTPUT_DIR"/instances_*.json 2>/dev/null | sed 's/^/    /'
        echo "  统计信息文件:"
        ls -la "$OUTPUT_DIR"/conversion_stats_*.json 2>/dev/null | sed 's/^/    /'
    fi
    
    echo ""
    echo -e "${GREEN}数据集已成功转换为COCO格式!${NC}"
    echo -e "${YELLOW}已自动创建:${NC}"
    echo "  - 训练集 (instances_train.json)"
    echo "  - 验证集 (instances_val.json) - 5000张图像"
    echo "  - 测试集 (instances_test.json) - 5000张图像"
else
    echo ""
    echo -e "${RED}转换失败! 请检查错误信息。${NC}"
    exit 1
fi