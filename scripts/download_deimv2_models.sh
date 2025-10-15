#!/bin/bash

# DEIMv2 模型下载脚本
# 此脚本用于下载DEIMv2预训练模型和日志文件

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
    
    if ! command -v gdown &> /dev/null; then
        print_warning "gdown 未安装，正在尝试安装..."
        if command -v pip &> /dev/null; then
            pip install gdown
        elif command -v pip3 &> /dev/null; then
            pip3 install gdown
        else
            print_error "需要安装 pip 来安装 gdown"
            print_info "请运行: pip install gdown"
            exit 1
        fi
    fi
    
    print_success "依赖检查完成"
}

# 从Google Drive URL提取文件ID
extract_file_id() {
    local url="$1"
    echo "$url" | sed -n 's/.*\/d\/\([a-zA-Z0-9_-]*\).*/\1/p'
}

# 下载文件函数
download_file() {
    local url="$1"
    local filename="$2"
    local output_path="$3"
    
    print_info "开始下载: $filename"
    print_info "保存到: $output_path"
    
    # 提取Google Drive文件ID
    local file_id=$(extract_file_id "$url")
    
    if [ -z "$file_id" ]; then
        print_error "无法从URL提取文件ID: $url"
        return 1
    fi
    
    # 使用gdown下载
    if gdown --id "$file_id" -O "$output_path" --no-check-certificate; then
        print_success "下载完成: $filename"
        return 0
    else
        print_error "下载失败: $filename"
        print_info "尝试使用备用方法..."
        
        # 备用方法：直接使用gdown的URL格式
        if gdown "$url" -O "$output_path" --fuzzy --no-check-certificate; then
            print_success "下载完成: $filename (备用方法)"
            return 0
        else
            print_error "所有下载方法都失败: $filename"
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

# 创建目录
create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        print_info "创建目录: $dir"
        mkdir -p "$dir"
    fi
}

# 主函数
main() {
    echo "=================================================="
    echo "          DEIMv2 模型下载脚本"
    echo "=================================================="
    echo
    echo "此脚本将下载DEIMv2预训练模型和日志文件。"
    echo
    
    # 检查依赖
    check_dependencies
    
    # 创建输出目录
    CKPT_DIR="./ckpts"
    LOG_DIR="./ckpts/logs"
    create_directory "$CKPT_DIR"
    create_directory "$LOG_DIR"
    
    # 定义模型信息 (模型名称 -> [checkpoint_url, log_url])
    declare -A models
    
    # DEIMv2 HGNetV2 模型
    models["deimv2_hgnetv2_atto"]="https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view?usp=sharing https://drive.google.com/file/d/1M7FLN8EeVHG02kegPN-Wxf_9BlkghZfj/view?usp=sharing"
    models["deimv2_hgnetv2_femto"]="https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view?usp=sharing https://drive.google.com/file/d/1_KWVfOr3bB5TMHTNOmDIAO-tZJmKB9-b/view?usp=sharing"
    models["deimv2_hgnetv2_pico"]="https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view?usp=sharing https://drive.google.com/file/d/1GwyWotYSKmFQdVN9k2MM6atogpbh0lo1/view?usp=sharing"
    models["deimv2_hgnetv2_n"]="https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view?usp=sharing https://drive.google.com/file/d/1QhYfRrUy8HrihD3OwOMJLC-ATr97GInV/view?usp=sharing"
    
    # DEIMv2 DINOv3 模型
    models["deimv2_dinov3_s"]="https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing https://drive.google.com/file/d/1ydA4lWiTYusV1s3WHq5jSxIq39oxy-Nf/view?usp=sharing"
    models["deimv2_dinov3_m"]="https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=sharing https://drive.google.com/file/d/1i05Q1-O9UH-2Vb52FpFJ4mBG523GUqJU/view?usp=sharing"
    models["deimv2_dinov3_l"]="https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view?usp=sharing https://drive.google.com/file/d/13mrQxyrf1kJ45Yd692UQwdb7lpGoqsiS/view?usp=sharing"
    models["deimv2_dinov3_x"]="https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view?usp=sharing https://drive.google.com/file/d/13QV0SwJw1wHl0xHWflZj1KstBUAovSsV/view?usp=drive_link"
    
    # 显示可用模型
    echo "可用模型:"
    echo "1. deimv2_hgnetv2_atto (AP: 23.8, 0.5M params)"
    echo "2. deimv2_hgnetv2_femto (AP: 31.0, 1.0M params)"
    echo "3. deimv2_hgnetv2_pico (AP: 38.5, 1.5M params)"
    echo "4. deimv2_hgnetv2_n (AP: 43.0, 3.6M params)"
    echo "5. deimv2_dinov3_s (AP: 50.9, 9.7M params)"
    echo "6. deimv2_dinov3_m (AP: 53.0, 18.1M params)"
    echo "7. deimv2_dinov3_l (AP: 56.0, 32.2M params)"
    echo "8. deimv2_dinov3_x (AP: 57.8, 50.3M params)"
    echo "9. 下载所有模型"
    echo "10. 仅下载checkpoint文件"
    echo "11. 仅下载日志文件"
    echo
    
    # 用户选择
    if [ $# -eq 0 ]; then
        read -p "请选择要下载的模型 (1-11): " choice
    else
        choice="$1"
    fi
    
    case $choice in
        1|2|3|4|5|6|7|8)
            # 下载单个模型
            model_names=("deimv2_hgnetv2_atto" "deimv2_hgnetv2_femto" "deimv2_hgnetv2_pico" "deimv2_hgnetv2_n" "deimv2_dinov3_s" "deimv2_dinov3_m" "deimv2_dinov3_l" "deimv2_dinov3_x")
            model_name="${model_names[$((choice-1))]}"
            urls=(${models[$model_name]})
            ckpt_url="${urls[0]}"
            log_url="${urls[1]}"
            
            print_info "下载模型: $model_name"
            
            # 下载checkpoint
            ckpt_file="$CKPT_DIR/${model_name}.pth"
            if [ -f "$ckpt_file" ]; then
                print_warning "Checkpoint文件已存在: $ckpt_file"
                read -p "是否重新下载? (y/N): " overwrite
                if [[ $overwrite =~ ^[Yy]$ ]]; then
                    download_file "$ckpt_url" "${model_name}.pth" "$ckpt_file"
                    verify_file "$ckpt_file" "${model_name}.pth"
                fi
            else
                download_file "$ckpt_url" "${model_name}.pth" "$ckpt_file"
                verify_file "$ckpt_file" "${model_name}.pth"
            fi
            
            # 下载日志
            log_file="$LOG_DIR/${model_name}.log"
            if [ -f "$log_file" ]; then
                print_warning "日志文件已存在: $log_file"
                read -p "是否重新下载? (y/N): " overwrite
                if [[ $overwrite =~ ^[Yy]$ ]]; then
                    download_file "$log_url" "${model_name}.log" "$log_file"
                    verify_file "$log_file" "${model_name}.log"
                fi
            else
                download_file "$log_url" "${model_name}.log" "$log_file"
                verify_file "$log_file" "${model_name}.log"
            fi
            ;;
        9)
            # 下载所有模型
            print_info "开始下载所有模型..."
            success_count=0
            total_count=$((${#models[@]} * 2))  # checkpoint + log
            
            for model_name in "${!models[@]}"; do
                urls=(${models[$model_name]})
                ckpt_url="${urls[0]}"
                log_url="${urls[1]}"
                
                print_info "处理模型: $model_name"
                
                # 下载checkpoint
                ckpt_file="$CKPT_DIR/${model_name}.pth"
                if [ -f "$ckpt_file" ]; then
                    print_warning "跳过已存在的文件: $ckpt_file"
                    success_count=$((success_count + 1))
                else
                    if download_file "$ckpt_url" "${model_name}.pth" "$ckpt_file" && verify_file "$ckpt_file" "${model_name}.pth"; then
                        success_count=$((success_count + 1))
                    fi
                fi
                
                # 下载日志
                log_file="$LOG_DIR/${model_name}.log"
                if [ -f "$log_file" ]; then
                    print_warning "跳过已存在的文件: $log_file"
                    success_count=$((success_count + 1))
                else
                    if download_file "$log_url" "${model_name}.log" "$log_file" && verify_file "$log_file" "${model_name}.log"; then
                        success_count=$((success_count + 1))
                    fi
                fi
                
                echo
            done
            
            print_info "下载完成: $success_count/$total_count 个文件成功"
            ;;
        10)
            # 仅下载checkpoint
            print_info "仅下载checkpoint文件..."
            success_count=0
            total_count=${#models[@]}
            
            for model_name in "${!models[@]}"; do
                urls=(${models[$model_name]})
                ckpt_url="${urls[0]}"
                
                ckpt_file="$CKPT_DIR/${model_name}.pth"
                if [ -f "$ckpt_file" ]; then
                    print_warning "跳过已存在的文件: $ckpt_file"
                    success_count=$((success_count + 1))
                else
                    if download_file "$ckpt_url" "${model_name}.pth" "$ckpt_file" && verify_file "$ckpt_file" "${model_name}.pth"; then
                        success_count=$((success_count + 1))
                    fi
                fi
            done
            
            print_info "下载完成: $success_count/$total_count 个checkpoint文件成功"
            ;;
        11)
            # 仅下载日志
            print_info "仅下载日志文件..."
            success_count=0
            total_count=${#models[@]}
            
            for model_name in "${!models[@]}"; do
                urls=(${models[$model_name]})
                log_url="${urls[1]}"
                
                log_file="$LOG_DIR/${model_name}.log"
                if [ -f "$log_file" ]; then
                    print_warning "跳过已存在的文件: $log_file"
                    success_count=$((success_count + 1))
                else
                    if download_file "$log_url" "${model_name}.log" "$log_file" && verify_file "$log_file" "${model_name}.log"; then
                        success_count=$((success_count + 1))
                    fi
                fi
            done
            
            print_info "下载完成: $success_count/$total_count 个日志文件成功"
            ;;
        *)
            print_error "无效选择"
            exit 1
            ;;
    esac
    
    echo
    echo "=================================================="
    print_success "下载任务完成!"
    echo "Checkpoint文件保存在: $CKPT_DIR"
    echo "日志文件保存在: $LOG_DIR"
    echo "=================================================="
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi