#!/bin/bash

# DEIMv2 DINOv3-X 自定义数据集测试脚本
# 基于README.md中的测试命令格式

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
CONFIG_FILE="configs/deimv2/deimv2_dinov3_x_custom.yml"
GPUS="0,1,2,3"
MASTER_PORT="7777"
NPROC_PER_NODE="4"
MODEL_PATH=""
OUTPUT_DIR=""

# 显示帮助信息
show_help() {
    echo -e "${BLUE}DEIMv2 DINOv3-X 自定义数据集测试脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config FILE       配置文件路径 (默认: configs/deimv2/deimv2_dinov3_x_custom.yml)"
    echo "  -r, --model PATH        模型检查点路径 (必需)"
    echo "  -g, --gpus GPUS         GPU设备列表 (默认: 0,1,2,3)"
    echo "  -p, --port PORT         主端口号 (默认: 7777)"
    echo "  -n, --nproc NUM         每个节点的进程数 (默认: 4)"
    echo "  -o, --output DIR        输出目录 (可选，覆盖配置文件中的设置)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -r model.pth                      # 使用默认参数测试"
    echo "  $0 -r model.pth -g 0,1 -n 2          # 使用2个GPU测试"
    echo "  $0 -r model.pth -o ./test_results    # 指定输出目录"
    echo ""
    echo "注意:"
    echo "  - 模型检查点路径是必需的参数"
    echo "  - 测试将使用验证集数据进行评估"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -n|--nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -o|--output)
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

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 必须指定模型检查点路径${NC}"
    echo "使用 '$0 --help' 查看帮助信息"
    exit 1
fi

# 验证参数
echo -e "${BLUE}=== DEIMv2 DINOv3-X 自定义数据集测试 ===${NC}"
echo -e "${YELLOW}测试参数:${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  模型路径: $MODEL_PATH"
echo "  GPU设备: $GPUS"
echo "  主端口: $MASTER_PORT"
echo "  进程数: $NPROC_PER_NODE"
[ -n "$OUTPUT_DIR" ] && echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型检查点文件不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 检查数据集是否存在
echo -e "${YELLOW}检查数据集...${NC}"
if [ ! -d "dataset/data" ]; then
    echo -e "${RED}错误: 图像数据目录不存在: dataset/data${NC}"
    exit 1
fi

if [ ! -f "dataset/coco_format/instances_val.json" ]; then
    echo -e "${RED}错误: 验证标注文件不存在: dataset/coco_format/instances_val.json${NC}"
    echo -e "${YELLOW}提示: 请先运行数据集转换脚本: ./scripts/convert_dataset_simple.sh${NC}"
    exit 1
fi

# 构建测试命令
TEST_CMD="CUDA_VISIBLE_DEVICES=$GPUS torchrun --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE train.py -c $CONFIG_FILE --test-only -r $MODEL_PATH"

# 添加可选参数
[ -n "$OUTPUT_DIR" ] && TEST_CMD="$TEST_CMD --output-dir $OUTPUT_DIR"

# 显示完整命令
echo -e "${GREEN}执行测试命令:${NC}"
echo "$TEST_CMD"
echo ""

# 确认执行
read -p "是否开始测试? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}测试已取消${NC}"
    exit 0
fi

# 执行测试
echo -e "${GREEN}开始测试...${NC}"
echo ""

eval $TEST_CMD

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 测试完成! ===${NC}"
    echo -e "${YELLOW}结果已保存到输出目录${NC}"
    
    # 尝试显示测试结果摘要
    if [ -n "$OUTPUT_DIR" ]; then
        RESULT_DIR="$OUTPUT_DIR"
    else
        RESULT_DIR="outputs/deimv2_dinov3_x_custom"
    fi
    
    echo -e "${YELLOW}查找测试结果文件...${NC}"
    if [ -d "$RESULT_DIR" ]; then
        echo "输出目录: $RESULT_DIR"
        # 查找可能的结果文件
        find "$RESULT_DIR" -name "*.json" -o -name "*.txt" -o -name "*.log" | head -5
    fi
else
    echo ""
    echo -e "${RED}测试失败! 请检查错误信息。${NC}"
    exit 1
fi