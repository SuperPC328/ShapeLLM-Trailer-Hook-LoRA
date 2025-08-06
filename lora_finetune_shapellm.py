#!/usr/bin/env python3
"""
ShapeLLM LoRA微调脚本

处理环境切换问题：
- ShapeLLM-Omni运行在shapeLLM环境
- llama-factory运行在base环境
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import shutil

class ShapeLLMLoRATrainer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "llamafactory_configs"
        self.data_dir = self.project_root / "training_data"
        self.output_dir = self.project_root / "lora_output"
        
        # 创建必要的目录
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def check_environments(self):
        """检查环境配置"""
        print("=== 检查环境配置 ===")
        
        # 检查当前环境
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"当前环境: {current_env}")
        
        # 检查llama-factory是否在base环境
        try:
            result = subprocess.run(['which', 'llamafactory-cli'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ 找到llamafactory-cli: {result.stdout.strip()}")
            else:
                print("✗ 未找到llamafactory-cli")
                return False
        except Exception as e:
            print(f"✗ 检查llamafactory-cli失败: {e}")
            return False
        
        return True
    
    def find_shapellm_model(self):
        """查找本地ShapeLLM模型"""
        print("=== 查找ShapeLLM模型 ===")
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        shapellm_dirs = [d for d in cache_dir.iterdir() if "ShapeLLM" in d.name]
        
        if shapellm_dirs:
            shapellm_dir = shapellm_dirs[0]
            snapshots_dir = shapellm_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    model_path = snapshots[0]
                    print(f"✓ 找到ShapeLLM模型: {model_path}")
                    return str(model_path)
        
        print("✗ 未找到本地ShapeLLM模型")
        return None
    
    def create_dataset_config(self, data_file):
        """创建数据集配置文件"""
        print("=== 创建数据集配置 ===")

        # 检查数据文件是否存在
        if not Path(data_file).exists():
            print(f"✗ 数据文件不存在: {data_file}")
            return False

        # 在当前项目目录创建data目录
        data_dir = self.project_root / "data"
        data_dir.mkdir(exist_ok=True)

        # 复制训练数据，使用原始文件名
        data_filename = Path(data_file).name
        target_data_file = data_dir / data_filename
        shutil.copy2(data_file, target_data_file)
        print(f"✓ 训练数据复制到: {target_data_file}")

        # 创建dataset_info.json - 使用LlamaFactory兼容格式
        dataset_info = {
            "shapellm_bdf": {
                "file_name": data_filename,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages"
                }
            }
        }

        dataset_info_file = data_dir / "dataset_info.json"
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print(f"✓ 数据集配置创建: {dataset_info_file}")
        return True
    
    def create_training_config(self, model_path):
        """创建训练配置文件 - 优化48GB显存使用和wandb集成"""
        print("=== 创建训练配置 ===")

        config = {
            # 模型和数据配置
            "model_name_or_path": model_path,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "all",
            "dataset": "shapellm_bdf",
            "template": "llama3",
            "cutoff_len": 4096,
            "max_samples": 1000,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,

            # 输出和日志配置
            "output_dir": str(self.output_dir),
            "logging_steps": 5,
            "save_steps": 100,
            "plot_loss": True,
            "overwrite_output_dir": True,

            # 显存优化配置 (保守设置)
            "per_device_train_batch_size": 1,  # 最小batch size
            "gradient_accumulation_steps": 4, # 增加梯度累积保持有效batch size
            "dataloader_num_workers": 2,       # 减少工作进程
            "dataloader_pin_memory": False,    # 禁用内存固定
            "max_grad_norm": 1.0,              # 梯度裁剪
            "gradient_checkpointing": True,    # 启用梯度检查点节省显存

            # 学习率和优化器配置
            "learning_rate": 2e-4,              # 稍微提高学习率
            "num_train_epochs": 10.0,            # 增加训练轮数
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 0.00000001,         # 1e-8 as float

            # 精度和性能配置
            "bf16": True,
            "tf32": True,                       # 启用TF32加速
            "ddp_timeout": 180000000,
            "include_num_input_tokens_seen": True,

            # LoRA配置 (保守设置节省显存)
            "lora_rank": 8,                     # 减少LoRA rank节省显存
            "lora_alpha": 16,                   # 相应调整alpha
            "lora_dropout": 0.05,
            "save_only_model": True,

            # 本地监控 (不使用wandb)
            "report_to": "none",  # 禁用wandb
            "run_name": "shapellm_bdf_lora_finetune",
            "logging_first_step": True,
            "save_total_limit": 3,
            "logging_dir": str(self.output_dir / "logs"),  # 本地日志目录
        }
        
        config_file = self.config_dir / "train_shapellm_lora.yaml"
        
        # 转换为YAML格式
        yaml_content = []
        for key, value in config.items():
            if isinstance(value, str):
                yaml_content.append(f"{key}: '{value}'")
            elif isinstance(value, bool):
                yaml_content.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, float):
                # 确保浮点数格式正确
                if value < 1e-6:
                    yaml_content.append(f"{key}: {value:.2e}")  # 科学计数法
                else:
                    yaml_content.append(f"{key}: {value}")
            else:
                yaml_content.append(f"{key}: {value}")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yaml_content))
        
        print(f"✓ 训练配置创建: {config_file}")
        return str(config_file)
    
    def run_lora_training(self, config_file):
        """运行LoRA训练"""
        print("=== 开始LoRA训练 ===")

        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # 确保使用GPU 0
        env['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告

        # 确保在base环境中运行
        cmd = [
            'llamafactory-cli', 'train', config_file
        ]

        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {self.project_root}")

        try:
            # 使用实时输出，设置正确的工作目录
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=str(self.project_root),  # 设置工作目录
                env=env
            )

            # 实时显示输出
            for line in process.stdout:
                print(line.rstrip())

            process.wait()

            if process.returncode == 0:
                print("✓ LoRA训练完成")
                return True
            else:
                print(f"✗ LoRA训练失败，返回码: {process.returncode}")
                return False

        except Exception as e:
            print(f"✗ 训练过程出错: {e}")
            return False
    
    def run_full_pipeline(self, data_file):
        """运行完整的LoRA微调流程"""
        print("=== ShapeLLM LoRA微调流程 ===\n")
        
        # 步骤1: 检查环境
        if not self.check_environments():
            return False
        
        # 步骤2: 查找模型
        model_path = self.find_shapellm_model()
        if not model_path:
            return False
        
        # 步骤3: 创建数据集配置
        if not self.create_dataset_config(data_file):
            return False
        
        # 步骤4: 创建训练配置
        config_file = self.create_training_config(model_path)
        if not config_file:
            return False
        
        # 步骤5: 运行训练
        if not self.run_lora_training(config_file):
            return False
        
        print("\n🎉 LoRA微调完成!")
        print(f"模型保存在: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description="ShapeLLM LoRA微调")
    parser.add_argument("--data_file", type=str, required=True,
                       help="训练数据文件路径")
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not Path(args.data_file).exists():
        print(f"✗ 训练数据文件不存在: {args.data_file}")
        sys.exit(1)
    
    # 运行LoRA微调
    trainer = ShapeLLMLoRATrainer()
    success = trainer.run_full_pipeline(args.data_file)
    
    if success:
        print("\n🎉 LoRA微调成功完成!")
        sys.exit(0)
    else:
        print("\n💥 LoRA微调失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
