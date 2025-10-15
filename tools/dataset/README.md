# 数据格式转换工具

本目录包含将自定义数据格式转换为COCO格式的工具脚本。

## 文件说明

- `convert_to_coco.py`: 主要的数据格式转换脚本
- `convert_example.py`: 使用示例脚本
- `README.md`: 本说明文档

## 支持的数据格式

### 输入格式

**目录结构：**
```
dataset/
├── data/           # 图像文件目录
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
└── label/          # 标注文件目录
    ├── 00000.json
    ├── 00001.json
    └── ...
```

**标注文件格式：**
```json
[
    {
        "id": "1",
        "type": "car",
        "box2d": [450.20944, 266.10209, 558.21807, 321.45953],
        "occluded": 0,
        "truncated": 0
    },
    {
        "id": "2",
        "type": "pedestrian",
        "box2d": [708.86265, 238.92889, 795.933, 290.0459],
        "occluded": 1,
        "truncated": 0
    }
]
```

**支持的类别（共14个）：**
| 序号 | 英文名称 | 中文名称 | 说明 |
|------|----------|----------|------|
| 1 | car | 小汽车 | 双轴小型车辆，包括7座及以下的轿车、SUV、皮卡、出租车、无人驾驶车辆、网约车 |
| 2 | truck | 货车 | 大中型货车，包括厢式货车、栅式货车、半挂车、平板运输货车、牵引车头、油罐车、混凝土搅拌车、消防车、环卫洒水车、泥头车、渣土车等 |
| 3 | construction_truck | 工程车辆 | 包括叉车、铲车、起重机、压路机、挖土机、推土机、吊车、矿车等工程车辆 |
| 4 | van | 厢式面包车 | 中型客运或货运面包车、救护车等 |
| 5 | bus | 巴士 | 大型客运车辆，包括校巴、旅游巴、客运班车、城市公交等 |
| 6 | bicycle | 两轮车 | 没有人骑着的两轮车，包含两轮摩托车、两轮电动车、两轮自行车 |
| 7 | cyclist | 两轮车骑行者 | 有人骑着的两轮车，包含两轮摩托车、两轮电动车、两轮自行车 |
| 8 | tricycle | 三轮车、三轮车骑行者 | 包含三轮摩托车、三轮电动车、三轮自行车等，包含有人或者无人的情况，若有人骑着需要把人也框住 |
| 9 | trolley | 手推车 | 包含手推车、婴儿车、轮椅。手推车不包含推车的人。婴儿车和轮椅如果有人坐着，应该标为一个整体 |
| 10 | pedestrian | 行人 | 包含各种姿态的行人，站立、蹲坐、躺着、静止与动态等，包含人的所有部位，手持任何其他东西不框进 |
| 11 | cone | 锥形桶、柱形桶 | 施工、交通意外中用于隔离的锥形桶、柱形防撞桶等 |
| 12 | barrier | 水马、栅栏 | 施工、交通意外中用于隔离的水马、栅栏等 |
| 13 | animal | 小动物 | 包含猫、狗等小动物 |
| 14 | other | 其他 | 车道上影响车辆正常驾驶的其他物体，例如路面碎片或落在车道上的树木等。不包括路上不影响驾驶的小物体，比如一小块垃圾、石头等 |

**字段说明：**
- `id`: 目标标识符（字符串类型）
- `type`: 目标类别名称（字符串类型）
- `box2d`: 边界框坐标 `[xmin, ymin, xmax, ymax]`（浮点型数组）
- `occluded`: 遮挡程度（整型）
  - 0: 0%~30%遮挡
  - 1: 30%~50%遮挡  
  - 2: 50%~70%遮挡
  - 3: 超过70%遮挡
- `truncated`: 截断状态（整型）
  - 0: 不截断
  - 1: 截断

### 输出格式

**COCO格式JSON文件：**
```json
{
    "images": [
        {
            "id": 1,
            "file_name": "00000.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [450.20944, 266.10209, 108.00863, 55.35744],
            "area": 5978.52,
            "iscrowd": 0,
            "segmentation": [],
            "occluded": 0,
            "truncated": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "car",
            "supercategory": "object",
            "chinese_name": "小汽车",
            "description": "双轴小型车辆，包括7座及以下的轿车、SUV、皮卡、出租车、无人驾驶车辆、网约车"
        },
        {
            "id": 2,
            "name": "truck",
            "supercategory": "object",
            "chinese_name": "货车",
            "description": "大中型货车，包括厢式货车、栅式货车、半挂车等"
        }
        // ... 其他9个类别
    ]
}
```

**类别特性：**
- 预定义了11个标准类别，每个类别都有固定的ID
- 包含中文名称和详细描述信息
- 如果数据中出现未预定义的类别，会自动分配新的ID（从12开始）
- 保留原始的遮挡度（occluded）和截断状态（truncated）信息

## 使用方法

### 方法1：使用示例脚本（推荐）

```bash
cd tools/dataset
python convert_example.py
```

### 方法2：直接使用转换脚本

```bash
cd tools/dataset
python convert_to_coco.py --data_dir dataset/data --label_dir dataset/label --output_dir dataset/coco_format --split_name train
```

### 方法3：在Python代码中使用

```python
from convert_to_coco import DatasetConverter

# 创建转换器
converter = DatasetConverter(
    data_dir="dataset/data",
    label_dir="dataset/label", 
    output_dir="dataset/coco_format"
)

# 执行转换
converter.convert_dataset()

# 保存结果
converter.save_coco_format("train")
```

## 参数说明

- `--data_dir`: 图像数据目录路径（默认：`dataset/data`）
- `--label_dir`: 标注文件目录路径（默认：`dataset/label`）
- `--output_dir`: 输出目录路径（默认：`dataset/coco_format`）
- `--split_name`: 数据集分割名称（默认：`train`，可选：`train`/`val`/`test`）

## 输出文件

转换完成后会生成以下文件：

1. `instances_{split_name}.json`: COCO格式的标注文件
2. `conversion_stats_{split_name}.json`: 转换统计信息

## 转换规则

1. **边界框格式转换**: `[xmin, ymin, xmax, ymax]` → `[x, y, width, height]`
2. **类别映射**: 自动为每个唯一的类别名称分配ID
3. **iscrowd标记**: 根据遮挡程度和截断状态自动设置
   - 遮挡程度≥3（>70%）或截断=1时，设置为crowd目标
4. **保留原始信息**: 在COCO标注中保留原始的`occluded`和`truncated`字段

## 配置文件更新

转换完成后，需要更新训练配置文件：

```yaml
# configs/dataset/custom_detection.yml
num_classes: 1  # 根据实际类别数量调整
remap_mscoco_category: False

train_dataloader:
  dataset:
    img_folder: dataset/data
    ann_file: dataset/coco_format/instances_train.json

val_dataloader:
  dataset:
    img_folder: dataset/data  
    ann_file: dataset/coco_format/instances_val.json
```

## 注意事项

1. 确保图像文件和标注文件的文件名（不含扩展名）一致
2. 支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
3. 标注文件必须是有效的JSON格式
4. 转换过程中会自动跳过无法读取的图像或标注文件
5. 建议在转换前备份原始数据

## 故障排除

### 常见问题

1. **找不到标注文件**
   - 检查图像文件名和标注文件名是否匹配
   - 确认标注文件扩展名为`.json`

2. **JSON解析错误**
   - 检查标注文件是否为有效的JSON格式
   - 使用JSON验证工具检查文件格式

3. **图像读取失败**
   - 检查图像文件是否损坏
   - 确认图像格式是否支持

4. **内存不足**
   - 对于大型数据集，考虑分批处理
   - 增加系统内存或使用更强大的机器

### 调试模式

在脚本中添加详细的日志输出来调试问题：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

如需添加其他功能，可以修改`DatasetConverter`类：

1. 支持其他标注格式
2. 添加数据增强
3. 支持分割标注
4. 添加数据验证功能