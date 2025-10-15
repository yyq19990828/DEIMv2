#!/bin/bash

# DINOv3 模型下载脚本
# 此脚本用于下载DINOv3预训练模型
# 使用此脚本即表示您同意DINOv3许可证条款和Meta隐私政策

set -e  # 遇到错误时退出

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

# 检查依赖
check_dependencies() {
    print_info "检查依赖项..."
    
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        print_error "需要安装 wget 或 curl 来下载文件"
        exit 1
    fi
    
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget"
        print_info "使用 wget 进行下载"
    else
        DOWNLOAD_CMD="curl"
        print_info "使用 curl 进行下载"
    fi
}

# 创建目录
create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        print_info "创建目录: $dir"
        mkdir -p "$dir"
    fi
}

# 下载文件函数
download_file() {
    local url="$1"
    local filename="$2"
    local output_path="$3"
    
    print_info "开始下载: $filename"
    print_info "保存到: $output_path"
    
    if [ "$DOWNLOAD_CMD" = "wget" ]; then
        if wget --progress=bar:force:noscroll -O "$output_path" "$url"; then
            print_success "下载完成: $filename"
            return 0
        else
            print_error "下载失败: $filename"
            return 1
        fi
    else
        if curl -L --progress-bar -o "$output_path" "$url"; then
            print_success "下载完成: $filename"
            return 0
        else
            print_error "下载失败: $filename"
            return 1
        fi
    fi
}

# 验证文件完整性
verify_file() {
    local filepath="$1"
    local filename="$2"
    
    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
        local filesize=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
        print_success "文件验证通过: $filename (大小: $filesize 字节)"
        return 0
    else
        print_error "文件验证失败: $filename"
        return 1
    fi
}

# 主函数
main() {
    echo "=================================================="
    echo "          DINOv3 模型下载脚本"
    echo "=================================================="
    echo
    echo "您即将下载DINOv3预训练模型。"
    echo "通过下载模型，您同意DINOv3许可证条款和Meta隐私政策。"
    echo
    
    # 检查依赖
    check_dependencies
    
    # 创建输出目录
    OUTPUT_DIR="./ckpts"
    create_directory "$OUTPUT_DIR"
    
    # 定义模型信息
    declare -A models
    models["dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]="https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicW50dWtpMHI0N3Zsd3R1dTJlcDZhZzFlIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjA2ODA3MTB9fX1dfQ__&Signature=trk6zjf2jZuRZXPMq3pAulOZVAW6G2QN9dACE5Rh30pMrQGc4Ywdn9JG9LK-sTnKrexJvywv7cksuzkeY6cObCyw5%7EAMy21Wq9pkyrpEkiWMXB9JYn7PU%7E3nMGPIAycxfuBGRuxI2dXWmmSiL%7EN9t3ljVx15kXslrXVHIloFYB0He5uwXfe-BLcbpcG0wIBPxhAwNdbzEEdQ2QUzQyj1PHikmBOHWNlu-nrDlNDyE5AeBaNRR8R7lur0n-13%7EOKa5tSbzymXBnr2g59fgK-7wFa2D464OcP8XzSBgDV339fDTCfORYneJNvKIpSRdrkOSCfXsXK%7EYbbFBoyZcicTnQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=762786973441312"
    models["dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"]="https://dinov3.llamameta.net/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicW50dWtpMHI0N3Zsd3R1dTJlcDZhZzFlIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjA2ODA3MTB9fX1dfQ__&Signature=trk6zjf2jZuRZXPMq3pAulOZVAW6G2QN9dACE5Rh30pMrQGc4Ywdn9JG9LK-sTnKrexJvywv7cksuzkeY6cObCyw5%7EAMy21Wq9pkyrpEkiWMXB9JYn7PU%7E3nMGPIAycxfuBGRuxI2dXWmmSiL%7EN9t3ljVx15kXslrXVHIloFYB0He5uwXfe-BLcbpcG0wIBPxhAwNdbzEEdQ2QUzQyj1PHikmBOHWNlu-nrDlNDyE5AeBaNRR8R7lur0n-13%7EOKa5tSbzymXBnr2g59fgK-7wFa2D464OcP8XzSBgDV339fDTCfORYneJNvKIpSRdrkOSCfXsXK%7EYbbFBoyZcicTnQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=762786973441312"
    
    # 显示可用模型
    echo "可用模型:"
    echo "1. dinov3_vits16_pretrain_lvd1689m-08c60483.pth (DINOv3 ViT-S/16)"
    echo "2. dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth (DINOv3 ViT-S/16+)"
    echo "3. 下载所有模型"
    echo
    
    # 用户选择
    if [ $# -eq 0 ]; then
        read -p "请选择要下载的模型 (1-3): " choice
    else
        choice="$1"
    fi
    
    case $choice in
        1)
            filename="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            url="${models[$filename]}"
            output_path="$OUTPUT_DIR/$filename"
            
            if [ -f "$output_path" ]; then
                print_warning "文件已存在: $filename"
                read -p "是否重新下载? (y/N): " overwrite
                if [[ ! $overwrite =~ ^[Yy]$ ]]; then
                    print_info "跳过下载"
                    exit 0
                fi
            fi
            
            download_file "$url" "$filename" "$output_path"
            verify_file "$output_path" "$filename"
            ;;
        2)
            filename="dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
            url="${models[$filename]}"
            output_path="$OUTPUT_DIR/$filename"
            
            if [ -f "$output_path" ]; then
                print_warning "文件已存在: $filename"
                read -p "是否重新下载? (y/N): " overwrite
                if [[ ! $overwrite =~ ^[Yy]$ ]]; then
                    print_info "跳过下载"
                    exit 0
                fi
            fi
            
            download_file "$url" "$filename" "$output_path"
            verify_file "$output_path" "$filename"
            ;;
        3)
            print_info "开始下载所有模型..."
            success_count=0
            total_count=${#models[@]}
            
            print_info "总共有 $total_count 个模型需要处理"
            
            for filename in "${!models[@]}"; do
                print_info "正在处理模型: $filename"
                url="${models[$filename]}"
                output_path="$OUTPUT_DIR/$filename"
                
                if [ -f "$output_path" ]; then
                    print_warning "文件已存在，跳过: $filename"
                    success_count=$((success_count + 1))
                    continue
                fi
                
                if download_file "$url" "$filename" "$output_path" && verify_file "$output_path" "$filename"; then
                    success_count=$((success_count + 1))
                else
                    print_error "下载失败: $filename"
                fi
                echo
            done
            
            print_info "下载完成: $success_count/$total_count 个文件成功"
            ;;
        *)
            print_error "无效选择"
            exit 1
            ;;
    esac
    
    echo
    echo "=================================================="
    print_success "下载任务完成!"
    echo "模型文件保存在: $OUTPUT_DIR"
    echo "=================================================="
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi