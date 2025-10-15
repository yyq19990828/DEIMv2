#!/bin/bash

# 数据集格式转换脚本
# 将自定义格式数据集转换为COCO格式

set -e  # 遇到错误时退出

# 脚本信息
SCRIPT_NAME="数据集格式转换工具"
VERSION="1.0.0"

# 默认参数
DEFAULT_DATA_DIR="dataset/data"
DEFAULT_LABEL_DIR="dataset/label"
DEFAULT_OUTPUT_DIR="dataset/coco_format"
DEFAULT_SPLIT_NAME="train"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
${SCRIPT_NAME} v${VERSION}

用法: $0 [选项]

选项:
    -d, --data-dir DIR      图像数据目录 (默认: ${DEFAULT_DATA_DIR})
    -l, --label-dir DIR     标注文件目录 (默认: ${DEFAULT_LABEL_DIR})
    -o, --output-dir DIR    输出目录 (默认: ${DEFAULT_OUTPUT_DIR})
    -s, --split-name NAME   数据集分割名称 (默认: ${DEFAULT_SPLIT_NAME})
    -h, --help              显示此帮助信息
    -v, --version           显示版本信息

示例:
    # 使用默认参数转换数据集
    $0

    # 指定自定义路径
    $0 -d /path/to/images -l /path/to/labels -o /path/to/output

    # 转换验证集
    $0 -s val

支持的类别 (共14个):
    1. car (小汽车)              8. tricycle (三轮车、三轮车骑行者)
    2. truck (货车)              9. trolley (手推车)
    3. construction_truck (工程车辆)  10. pedestrian (行人)
    4. van (厢式面包车)          11. cone (锥形桶、柱形桶)
    5. bus (巴士)               12. barrier (水马、栅栏)
    6. bicycle (两轮车)          13. animal (小动物)
    7. cyclist (两轮车骑行者)     14. other (其他)

EOF
}

# 显示版本信息
show_version() {
    echo "${SCRIPT_NAME} v${VERSION}"
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到，请确保已安装Python3"
        exit 1
    fi
    
    print_info "Python版本: $(python3 --version)"
}

# 检查必要的Python包
check_dependencies() {
    print_info "检查Python依赖包..."
    
    local missing_packages=()
    
    # 检查PIL/Pillow
    if ! python3 -c "from PIL import Image" 2>/dev/null; then
        missing_packages+=("Pillow")
    fi
    
    # 检查json (内置包，通常不会缺失)
    if ! python3 -c "import json" 2>/dev/null; then
        missing_packages+=("json")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "缺少以下Python包: ${missing_packages[*]}"
        print_info "请运行: pip3 install ${missing_packages[*]}"
        exit 1
    fi
    
    print_success "所有依赖包检查通过"
}

# 检查目录是否存在
check_directory() {
    local dir_path="$1"
    local dir_name="$2"
    
    if [ ! -d "$dir_path" ]; then
        print_error "${dir_name}目录不存在: $dir_path"
        return 1
    fi
    
    print_info "${dir_name}目录: $dir_path"
    return 0
}

# 检查转换脚本是否存在
check_converter_script() {
    local script_path="tools/dataset/convert_to_coco.py"
    
    if [ ! -f "$script_path" ]; then
        print_error "转换脚本不存在: $script_path"
        print_info "请确保在项目根目录运行此脚本"
        exit 1
    fi
    
    print_success "找到转换脚本: $script_path"
}

# 主转换函数
convert_dataset() {
    local data_dir="$1"
    local label_dir="$2"
    local output_dir="$3"
    local split_name="$4"
    
    print_info "开始数据集转换..."
    print_info "参数配置:"
    print_info "  图像目录: $data_dir"
    print_info "  标注目录: $label_dir"
    print_info "  输出目录: $output_dir"
    print_info "  分割名称: $split_name"
    
    # 执行转换
    python3 tools/dataset/convert_to_coco.py \
        --data_dir "$data_dir" \
        --label_dir "$label_dir" \
        --output_dir "$output_dir" \
        --split_name "$split_name"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "数据集转换完成！"
        print_info "输出文件:"
        print_info "  COCO格式文件: ${output_dir}/instances_${split_name}.json"
        print_info "  统计信息文件: ${output_dir}/conversion_stats_${split_name}.json"
        print_info ""
        print_info "接下来您可以："
        print_info "  1. 检查生成的COCO格式文件"
        print_info "  2. 使用 configs/dataset/converted_dataset.yml 配置文件"
        print_info "  3. 开始训练模型"
    else
        print_error "数据集转换失败，退出码: $exit_code"
        exit $exit_code
    fi
}

# 主函数
main() {
    # 解析命令行参数
    local data_dir="$DEFAULT_DATA_DIR"
    local label_dir="$DEFAULT_LABEL_DIR"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local split_name="$DEFAULT_SPLIT_NAME"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--data-dir)
                data_dir="$2"
                shift 2
                ;;
            -l|--label-dir)
                label_dir="$2"
                shift 2
                ;;
            -o|--output-dir)
                output_dir="$2"
                shift 2
                ;;
            -s|--split-name)
                split_name="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                print_info "使用 -h 或 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 显示脚本信息
    echo "=================================================="
    echo "  ${SCRIPT_NAME} v${VERSION}"
    echo "=================================================="
    echo ""
    
    # 执行检查
    check_python
    check_dependencies
    check_converter_script
    
    # 检查输入目录
    if ! check_directory "$data_dir" "图像数据"; then
        exit 1
    fi
    
    if ! check_directory "$label_dir" "标注文件"; then
        exit 1
    fi
    
    # 创建输出目录（如果不存在）
    if [ ! -d "$output_dir" ]; then
        print_info "创建输出目录: $output_dir"
        mkdir -p "$output_dir"
    fi
    
    echo ""
    
    # 执行转换
    convert_dataset "$data_dir" "$label_dir" "$output_dir" "$split_name"
}

# 运行主函数
main "$@"