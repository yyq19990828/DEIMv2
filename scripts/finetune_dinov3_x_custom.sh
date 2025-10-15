#!/bin/bash

# DEIMv2 DINOv3-X 自定义数据集微调脚本
# 基于README.md中的微调命令格式

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
USE_AMP="--use-amp"
SEED="0"
PRETRAINED_MODEL=""
OUTPUT_DIR=""

# 显示帮助信息
show_help() {
    echo -e "${BLUE}DEIMv2 DINOv3-X 自定义数据集微调脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config FILE       配置文件路径 (默认: configs/deimv2/deimv2_dinov3_x_custom.yml)"
    echo "  -t, --pretrained PATH   预训练模型路径 (必需)"
    echo "  -g, --gpus GPUS         GPU设备列表 (默认: 0,1,2,3)"
    echo "  -p, --port PORT         主端口号 (默认: 7777)"
    echo "  -n, --nproc NUM         每个节点的进程数 (默认: 4)"
    echo "  --no-amp               禁用自动混合精度训练"
    echo "  -s, --seed SEED         随机种子 (默认: 0)"
    echo "  -o, --output DIR        输出目录 (可选，覆盖配置文件中的设置)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -t pretrained_model.pth           # 使用默认参数微调"
    echo "  $0 -t model.pth -g 0,1 -n 2          # 使用2个GPU微调"
    echo "  $0 -t model.pth --no-amp             # 禁用混合精度微调"
    echo "  $0 -t model.pth -o ./finetune_outputs # 指定输出目录"
    echo ""
    echo "注意:"
    echo "  - 预训练模型路径是必需的参数"
    echo "  - 微调会在预训练模型基础上继续训练"
    echo "  - 建议使用较小的学习率进行微调"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--pretrained)
            PRETRAINED_MODEL="$2"
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
        --no-amp)
            USE_AMP=""
            shift
            ;;
        -s|--seed)
            SEED="$2"
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
if [ -z "$PRETRAINED_MODEL" ]; then
    echo -e "${RED}错误: 必须指定预训练模型路径${NC}"
    echo "使用 '$0 --help' 查看帮助信息"
    exit 1
fi

# 验证参数
echo -e "${BLUE}=== DEIMv2 DINOv3-X 自定义数据集微调 ===${NC}"
echo -e "${YELLOW}微调参数:${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  预训练模型: $PRETRAINED_MODEL"
echo "  GPU设备: $GPUS"
echo "  主端口: $MASTER_PORT"
echo "  进程数: $NPROC_PER_NODE"
echo "  混合精度: $([ -n "$USE_AMP" ] && echo "启用" || echo "禁用")"
echo "  随机种子: $SEED"
[ -n "$OUTPUT_DIR" ] && echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 检查预训练模型文件是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo -e "${RED}错误: 预训练模型文件不存在: $PRETRAINED_MODEL${NC}"
    exit 1
fi

# 检查数据集是否存在
echo -e "${YELLOW}检查数据集...${NC}"
if [ ! -d "dataset/data" ]; then
    echo -e "${RED}错误: 图像数据目录不存在: dataset/data${NC}"
    exit 1
fi

if [ ! -f "dataset/coco_format/instances_train.json" ]; then
    echo -e "${RED}错误: 训练标注文件不存在: dataset/coco_format/instances_train.json${NC}"
    echo -e "${YELLOW}提示: 请先运行数据集转换脚本: ./scripts/convert_dataset_simple.sh${NC}"
    exit 1
fi

if [ ! -f "dataset/coco_format/instances_val.json" ]; then
    echo -e "${RED}错误: 验证标注文件不存在: dataset/coco_format/instances_val.json${NC}"
    echo -e "${YELLOW}提示: 请先运行数据集转换脚本: ./scripts/convert_dataset_simple.sh${NC}"
    exit 1
fi

# 构建微调命令
FINETUNE_CMD="CUDA_VISIBLE_DEVICES=$GPUS torchrun --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE train.py -c $CONFIG_FILE"

# 添加可选参数
[ -n "$USE_AMP" ] && FINETUNE_CMD="$FINETUNE_CMD $USE_AMP"
FINETUNE_CMD="$FINETUNE_CMD --seed=$SEED"
FINETUNE_CMD="$FINETUNE_CMD -t $PRETRAINED_MODEL"
[ -n "$OUTPUT_DIR" ] && FINETUNE_CMD="$FINETUNE_CMD --output-dir $OUTPUT_DIR"

# 显示完整命令
echo -e "${GREEN}执行微调命令:${NC}"
echo "$FINETUNE_CMD"
echo ""

# 显示微调建议
echo -e "${YELLOW}微调建议:${NC}"
echo "  - 微调通常使用较小的学习率（比从头训练小10倍）"
echo "  - 建议减少训练轮数，避免过拟合"
echo "  - 可以考虑冻结部分层（如backbone的早期层）"
echo "  - 监控验证集性能，及时停止训练"
echo ""

# 确认执行
read -p "是否开始微调? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}微调已取消${NC}"
    exit 0
fi

# 执行微调
echo -e "${GREEN}开始微调...${NC}"
echo ""

eval $FINETUNE_CMD

# 检查微调结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== 微调完成! ===${NC}"
    echo -e "${YELLOW}输出目录: $([ -n "$OUTPUT_DIR" ] && echo "$OUTPUT_DIR" || echo "outputs/deimv2_dinov3_x_custom")${NC}"
    
    # 显示后续建议
    echo ""
    echo -e "${YELLOW}后续步骤建议:${NC}"
    echo "  1. 使用测试脚本评估微调后的模型性能"
    echo "  2. 比较微调前后的性能指标"
    echo "  3. 如果性能不理想，可以调整学习率或训练轮数重新微调"
else
    echo ""
    echo -e "${RED}微调失败! 请检查错误信息。${NC}"
    exit 1
fi