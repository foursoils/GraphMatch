import pandas as pd
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_yaml_config as load_config

def main():
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "data_prep.yaml")
    config = load_config(config_path)
    
    # 获取相对于当前脚本目录的基础路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 既然这是处理 minicheck 数据集的脚本，直接写死读取 minicheck.merge 的配置
    ds_config = config['minicheck']['merge']
    
    raw_dir = os.path.join(base_dir, ds_config['raw_dir'].strip('../'))
    processed_dir = os.path.join(base_dir, ds_config['processed_dir'].strip('../'))
    output_name = ds_config['merged_output_name']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"正在从目录加载文件: {raw_dir}")
    
    # 查找原始目录下的所有 parquet 文件
    pattern = os.path.join(raw_dir, "*.parquet")
    parquet_files = glob.glob(pattern)
    
    if not parquet_files:
        print(f"在 {raw_dir} 中未找到任何 parquet 文件！")
        return
        
    print(f"共找到 {len(parquet_files)} 个 parquet 文件。正在合并...")
    
    dfs = []
    for f in parquet_files:
        print(f"  - 正在读取 {os.path.basename(f)}")
        dfs.append(pd.read_parquet(f))
        
    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"合并完成，总行数: {len(merged_df)}")
    
    # 添加递增的字段 'id'
    print("正在分配顺序 'id' 字段...")
    # 将 id 添加到最前面（第0列）
    merged_df.insert(0, 'id', range(len(merged_df)))
    
    output_path = os.path.join(processed_dir, output_name)
    
    print(f"正在将合并后的数据集保存至 {output_path} ...")
    merged_df.to_parquet(output_path, index=False)
    
    print("\n✅ 数据预处理成功完成！")

if __name__ == '__main__':
    main()
