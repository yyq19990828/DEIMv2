# DEIMv2 DINOv3-X 自定义数据集训练指南

本指南介绍如何使用DEIMv2框架在自定义数据集上训练、测试和微调DINOv3-X模型。

## 📋 目录

- [前置准备](#前置准备)
- [数据集转换](#数据集转换)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型微调](#模型微调)
- [脚本说明](#脚本说明)
- [常见问题](#常见问题)

## 🚀 前置准备

### 1. 环境设置
```bash
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### 2. 数据集准备
确保您的数据集按以下结构组织：
```
dataset/
├── data/           # 图像文件
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
└── label/          # 标注文件（JSON格式）
    ├── 00000.json
    ├── 00001.json
    └── ...
```

### 3. 预训练权重下载
下载DINOv3预训练权重并放置到`./ckpts/`目录：
```bash
mkdir -p ckpts
# 下载 dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
# 放置到 ckpts/ 目录下
```

## 📊 数据集转换

首先将自定义格式的数据集转换为COCO格式：

```bash
# 转换数据集（自动创建训练集、验证集、测试集）
./scripts/convert_dataset_simple.sh

# 或者指定自定义路径
./scripts/convert_dataset_simple.sh -d your_data_dir -l your_label_dir -o your_output_dir
```

转换完成后，会在`dataset/coco_format/`目录下生成：
- `instances_train.json` - 训练集标注
- `instances_val.json` - 验证集标注（从训练集复制5000张）
- `instances_test.json` - 测试集标注（从训练集复制5000张）

## 🏋️ 模型训练

### 基础训练
```bash
# 使用默认参数训练
./scripts/train_dinov3_x_custom.sh

# 使用2个GPU训练
./scripts/train_dinov3_x_custom.sh -g 0,1 -n 2

# 禁用混合精度训练
./scripts/train_dinov3_x_custom.sh --no-amp

# 指定输出目录
./scripts/train_dinov3_x_custom.sh -o ./my_training_outputs
```

### 恢复训练
```bash
# 从检查点恢复训练
./scripts/train_dinov3_x_custom.sh -r outputs/model_epoch_10.pth
```

### 训练参数说明
- `-c, --config`: 配置文件路径
- `-g, --gpus`: GPU设备列表（如：0,1,2,3）
- `-n, --nproc`: 每个节点的进程数
- `--no-amp`: 禁用自动混合精度训练
- `-s, --seed`: 随机种子
- `-o, --output`: 输出目录
- `-r, --resume`: 恢复训练的检查点路径

## 🧪 模型测试

```bash
# 基础测试
./scripts/test_dinov3_x_custom.sh -r model.pth

# 使用2个GPU测试
./scripts/test_dinov3_x_custom.sh -r model.pth -g 0,1 -n 2

# 指定输出目录
./scripts/test_dinov3_x_custom.sh -r model.pth -o ./test_results
```

### 测试参数说明
- `-r, --model`: 模型检查点路径（必需）
- `-c, --config`: 配置文件路径
- `-g, --gpus`: GPU设备列表
- `-n, --nproc`: 每个节点的进程数
- `-o, --output`: 输出目录

## 🔧 模型微调

```bash
# 基础微调
./scripts/finetune_dinov3_x_custom.sh -t pretrained_model.pth

# 使用2个GPU微调
./scripts/finetune_dinov3_x_custom.sh -t model.pth -g 0,1 -n 2

# 禁用混合精度微调
./scripts/finetune_dinov3_x_custom.sh -t model.pth --no-amp

# 指定输出目录
./scripts/finetune_dinov3_x_custom.sh -t model.pth -o ./finetune_outputs
```

### 微调参数说明
- `-t, --pretrained`: 预训练模型路径（必需）
- `-c, --config`: 配置文件路径
- `-g, --gpus`: GPU设备列表
- `-n, --nproc`: 每个节点的进程数
- `--no-amp`: 禁用自动混合精度训练
- `-s, --seed`: 随机种子
- `-o, --output`: 输出目录

## 📝 脚本说明

### 配置文件
- `configs/deimv2/deimv2_dinov3_x_custom.yml`: DINOv3-X模型配置
- `configs/dataset/custom_dataset.yml`: 自定义数据集配置

### 核心脚本
1. **`convert_dataset_simple.sh`**: 数据集格式转换
2. **`train_dinov3_x_custom.sh`**: 模型训练
3. **`test_dinov3_x_custom.sh`**: 模型测试
4. **`finetune_dinov3_x_custom.sh`**: 模型微调

### 数据集配置
自定义数据集包含14个预定义类别：
1. car (小汽车)
2. truck (货车)
3. construction_truck (工程车辆)
4. van (厢式面包车)
5. bus (巴士)
6. bicycle (两轮车)
7. cyclist (两轮车骑行者)
8. tricycle (三轮车、三轮车骑行者)
9. trolley (手推车)
10. pedestrian (行人)
11. cone (锥形桶、柱形桶)
12. barrier (水马、栅栏)
13. animal (小动物)
14. other (其他)

## ❓ 常见问题

### Q1: 训练时出现CUDA内存不足错误
**A**: 尝试以下解决方案：
- 减少GPU数量：`-g 0,1 -n 2`
- 禁用混合精度：`--no-amp`
- 修改配置文件中的batch_size

### Q2: 数据集转换失败
**A**: 检查以下项目：
- 确保图像和标注文件一一对应
- 检查JSON标注文件格式是否正确
- 确保图像文件可以正常读取

### Q3: 模型训练收敛慢或不收敛
**A**: 尝试以下调整：
- 检查学习率设置
- 确认数据集质量和标注准确性
- 考虑使用预训练模型进行微调

### Q4: 如何监控训练进度
**A**: 训练日志会保存在输出目录中，可以使用tensorboard或查看日志文件监控训练进度。

### Q5: 如何调整训练参数
**A**: 修改配置文件`configs/deimv2/deimv2_dinov3_x_custom.yml`中的相关参数，如学习率、训练轮数等。

## 📞 技术支持

如果遇到问题，请：
1. 检查本文档的常见问题部分
2. 查看DEIMv2项目的GitHub Issues
3. 确保环境配置正确

## 🎯 完整工作流程示例

```bash
# 1. 转换数据集
./scripts/convert_dataset_simple.sh

# 2. 训练模型
./scripts/train_dinov3_x_custom.sh

# 3. 测试模型
./scripts/test_dinov3_x_custom.sh -r outputs/deimv2_dinov3_x_custom/model_final.pth

# 4. 微调模型（可选）
./scripts/finetune_dinov3_x_custom.sh -t pretrained_model.pth
```

祝您训练顺利！🚀