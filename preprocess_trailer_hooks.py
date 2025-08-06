#!/usr/bin/env python3
"""
ShapeLLM-Omni 拖车钩数据预处理脚本
完成从OBJ文件到训练数据的完整流程
"""

import os
import sys
import json
import hashlib
import random
import argparse
import pandas as pd
import numpy as np
import torch
import trimesh
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import tempfile

# 添加ShapeLLM-Omni路径
sys.path.append('ShapeLLM-Omni')
from trellis.models.sparse_structure_vqvae import VQVAE3D
from huggingface_hub import hf_hub_download

class TrailerHookPreprocessor:
    def __init__(self, mesh_dir="mesh_output", output_dir="processed_data", use_cache=True):
        self.mesh_dir = Path(mesh_dir)
        self.output_dir = Path(output_dir)
        self.use_cache = use_cache
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 拖车钩设计变体映射
        self.hook_variants = {
            "Hook1.obj": "original design",
            "Hook2.obj": "widened and thickened support arm", 
            "Hook3.obj": "reduced base plate size",
            "Hook4.obj": "extended base plate length",
            "Hook5.obj": "lengthened support arm neck",
            "Hook6.obj": "thickened base plate",
            "Hook7.obj": "increased support arm height"
        }
        
        # 3D编辑任务模板 (完全按照templates.txt)
        self.edit_templates = [
            "extended base plate for enhanced stability",
            "increased support arm height for better clearance",
            "lengthened support arm neck for improved reach",
            "reduced base plate size for space-saving installation",
            "thickened base plate for structural strength",
            "widened support arm for enhanced load capacity"
        ]
        
        # 加载任务模板
        self.load_templates()
        
        # 初始化3D VQ-VAE
        self.init_vqvae()
        
    def load_templates(self):
        """加载任务模板 - 完全按照templates.txt"""
        # Text-to-3D模板 (完整列表)
        self.text_to_3d_templates = [
            "Create a 3D mesh using the following description:",
            "Create a 3D object using the following description:",
            "Create a 3D assets using the following description:",
            "Create a 3D content using the following description:",
            "Please generate a 3D mesh based on the prompt I provided:",
            "Please generate a 3D object based on the prompt I provided:",
            "Please generate a 3D assets based on the prompt I provided:",
            "Please generate a 3D content based on the prompt I provided:",
            "Create a 3D mesh of",
            "Create a 3D object of",
            "Create a 3D assets of",
            "Create a 3D content of",
            "generate a 3D mesh of",
            "generate a 3D model of",
            "generate a 3D object of",
            "generate a 3D assets of",
            "generate a 3D content of",
            "Please create a 3D mesh using the following details:",
            "Please create a 3D object using the following details:",
            "Please create a 3D assets using the following details:",
            "Please create a 3D content using the following details:",
            "Please generate a 3D mesh based on the description I provided:",
            "Please generate a 3D object based on the description I provided:",
            "Please generate a 3D assets based on the description I provided:",
            "Please generate a 3D content based on the description I provided:"
        ]

        # 3D Caption模板 (完整列表)
        self.caption_templates = [
            'Can you give a brief overview of this 3D voxel?',
            'Give a quick overview of the object represented by this 3D mesh.',
            'How would you describe the 3D form shown in this 3D voxel?',
            'What kind of structure does this 3D obj depict?',
            'What can you infer about the object from this 3D voxel?',
            'Convey a summary of the 3D structure represented in this 3D assets.',
            'Give a concise interpretation of the 3D data presented here.',
            'Express in brief, what this 3D mesh is representing.',
            'Explain the object this 3D content depicts succinctly.',
            'Describe the object that this 3D voxel forms.',
            'What is the nature of the object this 3D mesh is representing?',
            'Summarize the 3D object briefly.',
            'Offer a summary of the 3D object illustrated by this voxel.',
            'Can you briefly outline the shape represented by these mesh?',
            'Provide a short explanation of this 3D structure.',
            'Characterize the object this 3D content is illustrating.',
            'How would you summarize this 3D data?',
            "Provide an outline of this 3D shape's characteristics.",
            "Present a compact account of this 3D object's key features.",
            'What object is this voxel rendering?',
            'What does this collection of mesh represent?',
            'What kind of object is depicted by this mesh?',
            'Deliver a quick description of the object represented here.',
            'Offer a succinct summary of this 3D object.',
            'Offer a clear and concise description of this 3D mesh object.',
            'Share a brief interpretation of this 3D assets.',
            'Could you delineate the form indicated by this mesh?',
            'What kind of object is illustrated by this collection of voxel?',
            'How would you interpret this 3D object?'
        ]

    def find_cached_model(self, repo_id, filename):
        """查找本地缓存的模型文件"""
        cache_base = ".cache/huggingface/hub"
        model_dir = f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = os.path.join(cache_base, model_dir, "snapshots")

        if os.path.exists(snapshots_dir):
            # 查找最新的快照目录
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # 使用第一个找到的快照（通常只有一个）
                snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
                model_path = os.path.join(snapshot_dir, filename)
                if os.path.exists(model_path):
                    return model_path
        return None

    def init_vqvae(self):
        """初始化3D VQ-VAE模型"""
        print("Loading 3D VQ-VAE model...")
        self.vqvae = VQVAE3D(num_embeddings=8192)
        self.vqvae.eval()

        # 根据设置决定是否使用缓存
        if self.use_cache:
            cached_path = self.find_cached_model("yejunliang23/3DVQVAE", "3DVQVAE.bin")
            if cached_path:
                print(f"✓ Using cached model weights: {cached_path}")
                filepath = cached_path
            else:
                print("⚠ Cache not found, downloading from HuggingFace...")
                filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin")
        else:
            print("⚠ Cache disabled, downloading from HuggingFace...")
            filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin")

        state_dict = torch.load(filepath, map_location="cpu")
        self.vqvae.load_state_dict(state_dict)
        self.vqvae = self.vqvae.to(self.device)
        print("✓ 3D VQ-VAE model loaded successfully")

    def get_file_hash(self, file_path):
        """计算文件SHA256哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    def convert_trimesh_to_open3d(self, trimesh_mesh):
        """转换trimesh到open3d格式"""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(trimesh_mesh.vertices, dtype=np.float64)
        )
        o3d_mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(trimesh_mesh.faces, dtype=np.int32)
        )
        return o3d_mesh

    def rotate_points(self, points, axis='x', angle_deg=90):
        """旋转点云"""
        angle_rad = np.deg2rad(angle_deg)
        if axis == 'x':
            R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
        elif axis == 'y':
            R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
        elif axis == 'z':
            R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        return points @ R.T

    def voxelize_mesh(self, obj_path):
        """体素化3D网格"""
        print(f"Voxelizing {obj_path}...")
        
        # 加载网格
        mesh = trimesh.load(obj_path, force='mesh')
        mesh = self.convert_trimesh_to_open3d(mesh)
        
        # 标准化顶点到[-0.5, 0.5]范围
        vertices = np.asarray(mesh.vertices)
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        vertices_normalized = (vertices - min_vals) / (max_vals - min_vals)
        vertices = vertices_normalized * 1.0 - 0.5
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # 创建体素网格
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh, voxel_size=1/64, 
            min_bound=(-0.5, -0.5, -0.5), 
            max_bound=(0.5, 0.5, 0.5)
        )
        
        # 提取体素坐标
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
        vertices = (vertices + 0.5) / 64 - 0.5
        voxel = self.rotate_points(vertices, axis='x', angle_deg=90)
        
        return voxel

    def encode_voxel_to_tokens(self, voxel_points):
        """将体素编码为token序列"""
        # 转换为稀疏张量格式
        coords = ((torch.from_numpy(voxel_points) + 0.5) * 64).int().contiguous()
        ss = torch.zeros(1, 1, 64, 64, 64, dtype=torch.long)
        ss[:, :, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        
        # 使用VQ-VAE编码
        with torch.no_grad():
            encoding_indices = self.vqvae.Encode(ss.to(dtype=torch.float32).to(self.device))
            tokens = encoding_indices[0].cpu().numpy().tolist()
        
        return tokens

    def tokens_to_mesh_string(self, tokens):
        """将token转换为mesh字符串格式"""
        mesh_str = "<mesh-start>"
        for token in tokens:
            mesh_str += f"<mesh{token}>"
        mesh_str += "<mesh-end>"
        return mesh_str

    def process_single_hook(self, hook_file):
        """处理单个拖车钩文件"""
        obj_path = self.mesh_dir / hook_file
        if not obj_path.exists():
            print(f"Warning: {obj_path} not found, skipping...")
            return None
            
        # 体素化
        voxel_points = self.voxelize_mesh(obj_path)
        
        # 编码为tokens
        tokens = self.encode_voxel_to_tokens(voxel_points)
        mesh_string = self.tokens_to_mesh_string(tokens)
        
        # 获取设计描述
        description = self.hook_variants[hook_file]
        
        return {
            'file': hook_file,
            'description': description,
            'mesh_string': mesh_string,
            'voxel_count': len(voxel_points)
        }

    def generate_training_data(self, processed_hooks):
        """生成训练数据"""
        training_data = []
        
        print("Generating training data...")
        
        # 1. Text-to-3D任务 (6模板 × 7变体 = 42条)
        # 格式: <prompt> → response: <mesh-start><mesh-xxxx>.....<mesh-end>
        for hook_data in processed_hooks:
            templates = random.sample(self.text_to_3d_templates, 6)
            for template in templates:
                if template.endswith(":"):
                    prompt = f"{template} a trailer hitch with {hook_data['description']}"
                else:
                    prompt = f"{template} a trailer hitch with {hook_data['description']}"

                training_data.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt
                        },
                        {
                            "from": "gpt",
                            "value": hook_data['mesh_string']
                        }
                    ],
                    "task_type": "text_to_3d",
                    "hook_variant": hook_data['file']
                })
        
        # 2. 3D Caption任务 (6模板 × 7变体 = 42条)
        # 格式: <mesh-start><mesh-xxxx>.....<mesh-end><prompt> → response: <description>
        for hook_data in processed_hooks:
            templates = random.sample(self.caption_templates, 6)
            for template in templates:
                # 正确格式: mesh_string + prompt
                prompt_with_mesh = f"{hook_data['mesh_string']}{template}"
                response = f"This 3D mesh represents a trailer hitch with {hook_data['description']}."

                training_data.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt_with_mesh
                        },
                        {
                            "from": "gpt",
                            "value": response
                        }
                    ],
                    "task_type": "3d_caption",
                    "hook_variant": hook_data['file']
                })
        
        # 3. 3D Edited任务 (1原始 × 6编辑 = 6条)
        # 格式: <mesh-start><mesh-xxxx>.....<mesh-end><prompt> → response: <mesh-start><mesh-xxxx>.....<mesh-end>
        original_hook = next(h for h in processed_hooks if h['file'] == 'Hook1.obj')
        edited_hooks = [h for h in processed_hooks if h['file'] != 'Hook1.obj']

        for i, edited_hook in enumerate(edited_hooks):
            edit_instruction = self.edit_templates[i]
            # 正确格式: 原始mesh + 编辑指令
            prompt_with_mesh = f"{original_hook['mesh_string']}{edit_instruction}"

            training_data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt_with_mesh
                    },
                    {
                        "from": "gpt",
                        "value": edited_hook['mesh_string']
                    }
                ],
                "task_type": "3d_edited",
                "original_hook": "Hook1.obj",
                "edited_hook": edited_hook['file'],
                "edit_instruction": edit_instruction
            })
        
        return training_data

    def run(self):
        """运行完整的预处理流程"""
        print("🚀 Starting trailer hook data preprocessing...")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理所有拖车钩文件
        processed_hooks = []
        for hook_file in self.hook_variants.keys():
            result = self.process_single_hook(hook_file)
            if result:
                processed_hooks.append(result)
        
        print(f"✓ Processed {len(processed_hooks)} hook variants")
        
        # 生成训练数据
        training_data = self.generate_training_data(processed_hooks)
        
        # 保存训练数据
        output_file = self.output_dir / "trailer_hook_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # 保存处理统计
        stats = {
            "total_samples": len(training_data),
            "text_to_3d_samples": len([d for d in training_data if d['task_type'] == 'text_to_3d']),
            "caption_samples": len([d for d in training_data if d['task_type'] == '3d_caption']),
            "edited_samples": len([d for d in training_data if d['task_type'] == '3d_edited']),
            "hook_variants_processed": len(processed_hooks),
            "processed_hooks": [h['file'] for h in processed_hooks]
        }
        
        stats_file = self.output_dir / "preprocessing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Preprocessing completed!")
        print(f"📊 Generated {stats['total_samples']} training samples")
        print(f"   - Text-to-3D: {stats['text_to_3d_samples']}")
        print(f"   - 3D Caption: {stats['caption_samples']}")
        print(f"   - 3D Edited: {stats['edited_samples']}")
        print(f"💾 Data saved to: {output_file}")
        print(f"📈 Stats saved to: {stats_file}")

    def save_voxel_visualization(self, processed_hooks):
        """保存体素可视化文件"""
        voxel_dir = self.output_dir / "voxels"
        voxel_dir.mkdir(exist_ok=True)

        for hook_data in processed_hooks:
            # 重新体素化以获取点云数据
            obj_path = self.mesh_dir / hook_data['file']
            voxel_points = self.voxelize_mesh(obj_path)

            # 保存为PLY格式
            ply_file = voxel_dir / f"{hook_data['file'].replace('.obj', '.ply')}"
            self.save_ply_from_array(voxel_points, ply_file)

    def save_ply_from_array(self, vertices, filename):
        """保存点云为PLY格式"""
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {vertices.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "end_header"
        ]
        with open(filename, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, vertices, fmt="%.6f")

    def validate_training_data(self, training_data):
        """验证训练数据格式"""
        print("Validating training data format...")

        required_keys = ["conversations", "task_type"]
        for i, sample in enumerate(training_data):
            # 检查必需字段
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Sample {i} missing required key: {key}")

            # 检查对话格式
            conversations = sample["conversations"]
            if len(conversations) != 2:
                raise ValueError(f"Sample {i} should have exactly 2 conversations")

            if conversations[0]["from"] != "human" or conversations[1]["from"] != "gpt":
                raise ValueError(f"Sample {i} has incorrect conversation format")

        print("✓ Training data format validation passed")

def main():
    parser = argparse.ArgumentParser(description="Trailer Hook Data Preprocessing")
    parser.add_argument("--mesh_dir", default="mesh_output", help="Directory containing OBJ files")
    parser.add_argument("--output_dir", default="processed_data", help="Output directory")
    parser.add_argument("--save_voxels", action="store_true", help="Save voxel PLY files")
    parser.add_argument("--validate", action="store_true", help="Validate training data format")
    parser.add_argument("--no_cache", action="store_true", help="Disable using cached model weights")

    args = parser.parse_args()

    try:
        preprocessor = TrailerHookPreprocessor(args.mesh_dir, args.output_dir, use_cache=not args.no_cache)

        # 检查输入文件
        missing_files = []
        for hook_file in preprocessor.hook_variants.keys():
            if not (Path(args.mesh_dir) / hook_file).exists():
                missing_files.append(hook_file)

        if missing_files:
            print(f"❌ Missing hook files: {missing_files}")
            print(f"Please ensure all hook files are in {args.mesh_dir}/")
            return

        preprocessor.run()

        # 可选功能
        if args.save_voxels:
            print("Saving voxel visualizations...")
            # 需要重新处理以获取体素数据
            processed_hooks = []
            for hook_file in preprocessor.hook_variants.keys():
                result = preprocessor.process_single_hook(hook_file)
                if result:
                    processed_hooks.append(result)
            preprocessor.save_voxel_visualization(processed_hooks)
            print("✓ Voxel PLY files saved")

        if args.validate:
            # 加载并验证生成的训练数据
            output_file = Path(args.output_dir) / "trailer_hook_training_data.json"
            with open(output_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            preprocessor.validate_training_data(training_data)

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
