# ShapeLLM-Omni 拖车钩设计LoRA微调项目

基于ShapeLLM-Omni模型，使用7个拖车钩设计变体进行LoRA微调，采用多模板数据增强策略。

## 🎯 项目概述

**参考论文**: [ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding](https://arxiv.org/html/2506.01853v1)

**核心功能**:
- 7个拖车钩设计变体的3D生成和理解
- 多任务训练：Text-to-3D、3D-caption、3D-edited
- LoRA微调优化，专门针对拖车钩设计

**数据规模**: 90条训练样本
- Text-to-3D: 42条 (6模板 × 7变体)
- 3D-caption: 42条 (6模板 × 7变体)
- 3D-edited: 6条 (1原始 × 6编辑)

**拖车钩设计变体**:
- Hook1: 原始设计
- Hook2: 加宽加厚支撑臂
- Hook3: 缩小底板
- Hook4: 延长底板长度
- Hook5: 延长支撑臂颈部
- Hook6: 加厚底板
- Hook7: 增加支撑臂高度


## � 快速开始

### 环境要求
- Python 3.10+
- CUDA 11.8+ (推荐)
- 16GB+ RAM
- 8GB+ GPU内存

### 安装依赖
```bash
# 安装ShapeLLM-Omni依赖
cd ShapeLLM-Omni
pip install -r requirements.txt

# 返回项目根目录
cd ..
```

### 环境检查
```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查GPU内存
nvidia-smi
```

## 📊 数据处理流程

### 步骤1: 准备数据
将BDF文件放入`bdf_data/`目录，按以下结构组织：
```
bdf_data/
├── orginal/                    # 原始设计
├── Extend the base plate lengthwise/
├── Increase the overall height of the support arm/
├── Lengthen the neck of the support arm/
├── Shrink the base plate/
├── Thicken the base plate/
└── Widen and thicken the support arm/
```

### 步骤2: BDF到Mesh转换
```bash
python bdf_to_mesh_preprocessor.py --input_dir bdf_data --output_dir mesh_output
```

### 步骤3: 生成训练数据
```bash
python preprocess_trailer_hooks.py --mesh_dir mesh_output --output_dir processed_data
```
## 🔧 数据预处理详解

### 预处理脚本功能
`preprocess_trailer_hooks.py`提供完整的数据预处理流水线：

**核心功能**:
- 体素化：将OBJ文件转换为64³体素表示
- VQ-VAE编码：使用预训练3D VQ-VAE将体素编码为token序列
- 多任务数据生成：自动生成三种任务的训练样本
- 模板随机化：每个样本使用随机选择的提示模板

**使用方法**:
```bash
# 基本使用
python preprocess_trailer_hooks.py

# 完整功能
python preprocess_trailer_hooks.py \
    --mesh_dir mesh_output \
    --output_dir processed_data \
    --save_voxels \
    --validate
```

**输出文件**:
- `trailer_hook_training_data.json` - 90条训练样本
- `preprocessing_stats.json` - 处理统计信息
- `voxels/*.ply` - 体素可视化文件（可选）

### 任务格式说明

**Text-to-3D**: `<prompt>` → `<mesh-start><mesh-tokens><mesh-end>`
```json
{
  "conversations": [
    {"from": "human", "value": "Create a 3D mesh of a trailer hitch with original design"},
    {"from": "gpt", "value": "<mesh-start><mesh0><mesh1>...<mesh-end>"}
  ]
}
```

**3D-caption**: `<mesh-tokens><prompt>` → `<description>`
```json
{
  "conversations": [
    {"from": "human", "value": "<mesh-start>...<mesh-end>Give a quick overview of this 3D mesh."},
    {"from": "gpt", "value": "This 3D mesh represents a trailer hitch with original design."}
  ]
}
```

**3D-edited**: `<original-mesh><edit-instruction>` → `<edited-mesh>`
```json
{
  "conversations": [
    {"from": "human", "value": "<mesh-start>...<mesh-end>extended base plate for enhanced stability"},
    {"from": "gpt", "value": "<mesh-start><edited-tokens><mesh-end>"}
  ]
}
```

## 🎯 LoRA微调

### 微调脚本
```bash
python lora_finetune_shapellm.py --data_file processed_data/trailer_hook_training_data.json
```

### 配置文件
训练参数在`llamafactory_configs/train_shapellm_lora.yaml`中配置。

### 输出
微调后的LoRA适配器保存在`lora_output/`目录。

## 📁 项目结构

```
ShapeLLM-Trailer-Hook-LoRA/
├── README.md                           # 项目文档
├── preprocess_trailer_hooks.py         # 主预处理脚本
├── bdf_to_mesh_preprocessor.py         # BDF转换脚本
├── lora_finetune_shapellm.py          # LoRA微调脚本
├── llamafactory_configs/               # 训练配置
│   └── train_shapellm_lora.yaml
├── ShapeLLM-Omni/                     # 原始ShapeLLM项目
├── bdf_data/                          # BDF数据目录（空）
├── mesh_output/                       # Hook OBJ文件目录（空）
├── processed_data/                    # 预处理输出目录（空）
├── training_data/                     # 训练数据目录（空）
├── lora_output/                       # LoRA输出目录（空）
└── data/                             # 通用数据目录（空）
```

## ⚠️ 注意事项

### 系统要求
- **GPU内存**: 建议8GB+用于VQ-VAE推理
- **系统内存**: 建议16GB+用于网格处理
- **存储空间**: 约100MB用于输出文件

### 缓存优化
脚本会自动使用HuggingFace缓存的模型权重，避免重复下载：
```bash
# 强制重新下载（如果需要）
python preprocess_trailer_hooks.py --no_cache
```

### 常见问题
1. **CUDA内存不足**: 使用`export CUDA_VISIBLE_DEVICES=""`强制使用CPU
2. **OBJ文件加载失败**: 检查文件格式和路径
3. **体素化结果为空**: 确认网格是封闭的且大小合适

## 📚 参考资料

- [ShapeLLM-Omni论文](https://arxiv.org/html/2506.01853v1)
- [TRELLIS项目](https://github.com/microsoft/TRELLIS)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

## 📄 许可证

本项目遵循MIT许可证。

