# 模型下载脚本使用说明

## 概述

本目录包含两个下载脚本：
1. `download.sh` - 用于下载DINOv3预训练模型文件
2. `download_deimv2_models.sh` - 用于下载DEIMv2预训练模型和日志文件

## DINOv3 模型下载 (download.sh)

这个脚本用于下载DINOv3预训练模型文件。通过使用此脚本，您同意DINOv3许可证条款和Meta隐私政策。

## 系统要求

- Linux/macOS/Windows (with WSL)
- `wget` 或 `curl` 命令行工具
- 足够的磁盘空间（每个模型约80-90MB）

## 使用方法

### 1. 交互式下载

直接运行脚本，然后根据提示选择要下载的模型：

```bash
./download.sh
```

### 2. 命令行参数下载

您也可以直接指定要下载的模型：

```bash
# 下载第一个模型 (DINOv3 ViT-S/16)
./download.sh 1

# 下载第二个模型 (DINOv3 ViT-S/16+)
./download.sh 2

# 下载所有模型
./download.sh 3
```

## 可用模型

1. **dinov3_vits16_pretrain_lvd1689m-08c60483.pth**
   - DINOv3 ViT-S/16 模型
   - 大小：约83MB

2. **dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth**
   - DINOv3 ViT-S/16+ 模型
   - 大小：约83MB

## 功能特性

- ✅ 自动检测下载工具（wget/curl）
- ✅ 进度条显示下载进度
- ✅ 文件完整性验证
- ✅ 重复下载检测和确认
- ✅ 彩色输出和错误处理
- ✅ 自动创建目录结构

## 故障排除

### 下载失败
如果下载失败，请检查：
1. 网络连接是否正常
2. 是否有足够的磁盘空间
3. 下载链接是否仍然有效

### 权限问题
如果遇到权限问题，请确保脚本有执行权限：
```bash
chmod +x download.sh
```

### 依赖缺失
如果系统提示缺少wget或curl，请安装：

**Ubuntu/Debian:**
```bash
sudo apt-get install wget curl
```

**CentOS/RHEL:**
```bash
sudo yum install wget curl
```

**macOS:**
```bash
brew install wget curl
```

## 许可证

使用此脚本下载的模型受DINOv3许可证约束。请确保您已阅读并同意相关条款。

## 支持

如果遇到问题，请检查：
1. 网络连接
2. 磁盘空间
3. 系统依赖

---

## DEIMv2 模型下载 (download_deimv2_models.sh)

这个脚本用于下载DEIMv2预训练模型的checkpoint和训练日志文件。

### 系统要求

- Linux/macOS/Windows (with WSL)
- Python 和 pip
- `gdown` 工具（脚本会自动安装）

### 使用方法

#### 1. 交互式下载

```bash
./download_deimv2_models.sh
```

#### 2. 命令行参数下载

```bash
# 下载特定模型 (1-8)
./download_deimv2_models.sh 1  # Atto模型
./download_deimv2_models.sh 8  # X模型

# 下载所有模型
./download_deimv2_models.sh 9

# 仅下载checkpoint文件
./download_deimv2_models.sh 10

# 仅下载日志文件
./download_deimv2_models.sh 11
```

### 可用模型

| 模型 | AP | 参数量 | GFLOPs | 延迟(ms) |
|:---:|:---:|:---:|:---:|:---:|
| **Atto** | 23.8 | 0.5M | 0.8 | 1.10 |
| **Femto** | 31.0 | 1.0M | 1.7 | 1.45 |
| **Pico** | 38.5 | 1.5M | 5.2 | 2.13 |
| **N** | 43.0 | 3.6M | 6.8 | 2.32 |
| **S** | 50.9 | 9.7M | 25.6 | 5.78 |
| **M** | 53.0 | 18.1M | 52.2 | 8.80 |
| **L** | 56.0 | 32.2M | 96.7 | 10.47 |
| **X** | 57.8 | 50.3M | 151.6 | 13.75 |

### 文件结构

下载完成后，文件将保存在以下位置：
```
ckpts/
├── deimv2_hgnetv2_atto.pth
├── deimv2_hgnetv2_femto.pth
├── deimv2_hgnetv2_pico.pth
├── deimv2_hgnetv2_n.pth
├── deimv2_dinov3_s.pth
├── deimv2_dinov3_m.pth
├── deimv2_dinov3_l.pth
├── deimv2_dinov3_x.pth
└── logs/
    ├── deimv2_hgnetv2_atto.log
    ├── deimv2_hgnetv2_femto.log
    ├── deimv2_hgnetv2_pico.log
    ├── deimv2_hgnetv2_n.log
    ├── deimv2_dinov3_s.log
    ├── deimv2_dinov3_m.log
    ├── deimv2_dinov3_l.log
    └── deimv2_dinov3_x.log
```

### 功能特性

- ✅ 自动安装依赖 (gdown)
- ✅ 支持Google Drive大文件下载
- ✅ 进度显示和错误处理
- ✅ 文件完整性验证
- ✅ 重复下载检测
- ✅ 分类下载（仅checkpoint或仅日志）
- ✅ 彩色输出和详细日志

### 故障排除

#### gdown安装失败
如果gdown自动安装失败，请手动安装：
```bash
pip install gdown
# 或
pip3 install gdown
```

#### Google Drive下载限制
如果遇到Google Drive下载限制，请：
1. 等待一段时间后重试
2. 使用不同的网络连接
3. 手动从Google Drive下载

---

**注意：**
- DINOv3下载链接包含时间戳和签名，可能会过期。如果链接失效，请获取新的下载链接。
- DEIMv2模型文件较大，请确保有足够的磁盘空间和稳定的网络连接。