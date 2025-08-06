#!/usr/bin/env python3
"""
BDF文件到3D Mesh的预处理脚本
专门为ShapeLLM-Omni项目设计，只提取几何信息（节点坐标和元素连接关系）
"""

import os
import sys
import json
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import argparse
from pyNastran.bdf.bdf import BDF

class BDFGeometryExtractor:
    """
    从BDF文件中提取几何信息的类
    只关注节点坐标和元素连接关系，忽略材料、载荷、约束等信息
    """
    
    def __init__(self):
        self.logger_enabled = True
    
    def log(self, message):
        """简单的日志输出"""
        if self.logger_enabled:
            print(f"[BDF几何提取] {message}")
    
    def extract_geometry_from_bdf(self, bdf_path):
        """
        从BDF文件中提取几何信息
        
        参数:
            bdf_path (str): BDF文件路径
            
        返回:
            dict: 包含节点和元素信息的字典
        """
        self.log(f"开始处理BDF文件: {bdf_path}")
        
        try:
            # 读取BDF文件
            bdf = BDF()
            bdf.read_bdf(bdf_path, xref=False, punch=False, validate=False)
            
            # 提取节点信息
            nodes = {}
            for node_id, node in bdf.nodes.items():
                # 获取节点坐标
                position = node.get_position()
                nodes[node_id] = {
                    'id': node_id,
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                }
            
            self.log(f"提取了 {len(nodes)} 个节点")
            
            # 提取元素信息（只关注几何连接关系）
            elements = {}
            supported_element_types = ['CTETRA', 'CHEXA', 'CPENTA', 'CTRIA3', 'CQUAD4', 'CBAR', 'CBEAM']
            
            for elem_id, elem in bdf.elements.items():
                elem_type = elem.type
                
                # 只处理支持的元素类型
                if elem_type in supported_element_types:
                    # 获取元素的节点连接关系
                    node_ids = [int(nid) for nid in elem.node_ids if nid is not None]
                    
                    elements[elem_id] = {
                        'id': elem_id,
                        'type': elem_type,
                        'nodes': node_ids
                    }
            
            self.log(f"提取了 {len(elements)} 个有效元素")
            
            # 统计元素类型
            element_type_count = {}
            for elem in elements.values():
                elem_type = elem['type']
                element_type_count[elem_type] = element_type_count.get(elem_type, 0) + 1
            
            self.log(f"元素类型统计: {element_type_count}")
            
            return {
                'nodes': nodes,
                'elements': elements,
                'metadata': {
                    'source_file': bdf_path,
                    'num_nodes': len(nodes),
                    'num_elements': len(elements),
                    'element_types': element_type_count
                }
            }
            
        except Exception as e:
            self.log(f"处理BDF文件时出错: {str(e)}")
            return None
    
    def geometry_to_mesh(self, geometry_data):
        """
        将几何数据转换为trimesh对象
        
        参数:
            geometry_data (dict): 几何数据
            
        返回:
            trimesh.Trimesh: 网格对象
        """
        if not geometry_data:
            return None
        
        nodes = geometry_data['nodes']
        elements = geometry_data['elements']
        
        # 创建顶点数组
        vertices = []
        node_id_to_index = {}
        
        for i, (node_id, node_data) in enumerate(nodes.items()):
            vertices.append([node_data['x'], node_data['y'], node_data['z']])
            node_id_to_index[node_id] = i
        
        vertices = np.array(vertices)
        
        # 创建面数组（只处理三角形和四边形面元素）
        faces = []
        
        for elem_data in elements.values():
            elem_type = elem_data['type']
            node_ids = elem_data['nodes']
            
            # 将节点ID转换为顶点索引
            try:
                indices = [node_id_to_index[nid] for nid in node_ids]
            except KeyError as e:
                self.log(f"警告: 元素 {elem_data['id']} 引用了不存在的节点 {e}")
                continue
            
            # 根据元素类型创建面
            if elem_type == 'CTRIA3' and len(indices) == 3:
                # 三角形
                faces.append(indices)
            elif elem_type == 'CQUAD4' and len(indices) == 4:
                # 四边形，分解为两个三角形
                faces.append([indices[0], indices[1], indices[2]])
                faces.append([indices[0], indices[2], indices[3]])
            elif elem_type in ['CTETRA'] and len(indices) == 4:
                # 四面体，创建四个三角形面
                faces.extend([
                    [indices[0], indices[1], indices[2]],
                    [indices[0], indices[1], indices[3]],
                    [indices[0], indices[2], indices[3]],
                    [indices[1], indices[2], indices[3]]
                ])
            elif elem_type in ['CHEXA'] and len(indices) == 8:
                # 六面体，创建12个三角形面（每个面分解为2个三角形）
                # 定义六面体的6个面
                hex_faces = [
                    [0, 1, 2, 3],  # 底面
                    [4, 7, 6, 5],  # 顶面
                    [0, 4, 5, 1],  # 前面
                    [2, 6, 7, 3],  # 后面
                    [0, 3, 7, 4],  # 左面
                    [1, 5, 6, 2]   # 右面
                ]
                
                for face in hex_faces:
                    face_indices = [indices[i] for i in face]
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
        
        if not faces:
            self.log("警告: 没有找到可用的面元素，创建点云")
            # 如果没有面，创建一个简单的凸包
            try:
                from scipy.spatial import ConvexHull
                if len(vertices) >= 4:
                    hull = ConvexHull(vertices)
                    faces = hull.simplices.tolist()
                else:
                    self.log("顶点数量不足，无法创建凸包")
                    return None
            except ImportError:
                self.log("scipy未安装，无法创建凸包")
                return None
        
        faces = np.array(faces)
        
        try:
            # 创建trimesh对象
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # 基本清理
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            self.log(f"创建网格: {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")
            
            return mesh
            
        except Exception as e:
            self.log(f"创建网格时出错: {str(e)}")
            return None
    
    def save_mesh(self, mesh, output_path, format='obj'):
        """
        保存网格到文件
        
        参数:
            mesh (trimesh.Trimesh): 网格对象
            output_path (str): 输出路径
            format (str): 输出格式 ('obj', 'ply', 'stl')
        """
        if mesh is None:
            self.log("网格为空，无法保存")
            return False
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存网格
            mesh.export(output_path)
            self.log(f"网格已保存到: {output_path}")
            return True
            
        except Exception as e:
            self.log(f"保存网格时出错: {str(e)}")
            return False

def batch_process_bdf_files(input_dir, output_dir, file_pattern="*.bdf"):
    """
    批量处理BDF文件
    
    参数:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        file_pattern (str): 文件匹配模式
    """
    extractor = BDFGeometryExtractor()
    
    # 查找所有BDF文件（递归搜索）
    input_path = Path(input_dir)
    bdf_files = list(input_path.rglob(file_pattern))
    
    if not bdf_files:
        print(f"在 {input_dir} 中没有找到匹配 {file_pattern} 的文件")
        return
    
    print(f"找到 {len(bdf_files)} 个BDF文件")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理统计
    success_count = 0
    failed_files = []
    
    # 批量处理
    for bdf_file in tqdm(bdf_files, desc="处理BDF文件"):
        try:
            # 提取几何信息
            geometry_data = extractor.extract_geometry_from_bdf(str(bdf_file))
            
            if geometry_data is None:
                failed_files.append(str(bdf_file))
                continue
            
            # 转换为网格
            mesh = extractor.geometry_to_mesh(geometry_data)
            
            if mesh is None:
                failed_files.append(str(bdf_file))
                continue
            
            # 保存网格文件
            output_file = output_path / f"{bdf_file.stem}.obj"
            if extractor.save_mesh(mesh, str(output_file)):
                success_count += 1
                
                # 保存几何数据的JSON文件（用于调试）
                json_file = output_path / f"{bdf_file.stem}_geometry.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(geometry_data, f, indent=2, ensure_ascii=False)
            else:
                failed_files.append(str(bdf_file))
                
        except Exception as e:
            print(f"处理文件 {bdf_file} 时出错: {str(e)}")
            failed_files.append(str(bdf_file))
    
    # 输出处理结果
    print(f"\n处理完成:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print(f"  失败的文件:")
        for failed_file in failed_files:
            print(f"    - {failed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDF文件到3D Mesh的预处理脚本")
    parser.add_argument("--input_dir", type=str, default="bdf_data", 
                       help="输入BDF文件目录")
    parser.add_argument("--output_dir", type=str, default="mesh_output", 
                       help="输出网格文件目录")
    parser.add_argument("--pattern", type=str, default="*.bdf", 
                       help="BDF文件匹配模式")
    
    args = parser.parse_args()
    
    print("=== BDF到3D Mesh预处理脚本 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"文件模式: {args.pattern}")
    print()
    
    batch_process_bdf_files(args.input_dir, args.output_dir, args.pattern)
