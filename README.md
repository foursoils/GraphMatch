# GraphMatch: Graph-Augmented Cross-Attention for Hallucination Detection in Large Language Models

本项目的文件结构如下：

```text
GraphMatch/
├── ablation/                  # 消融实验代码
│   ├── check/                 # 消融检查模型 (gnn.py, graphcheck.py)
│   └── kg/                    # 知识图谱消融主脚本
├── appendix/                  # 附录 (token 计数分析等)
├── configs/                   # 配置文件目录
│   ├── ablation.yaml
│   ├── graph_gen.yaml
│   └── hallu_detect.yaml
├── contrast_experiment/       # 对比实验代码
│   ├── detect_main.py
│   └── detect_retry.py
├── cot_generation/            # 思维链 (CoT) 生成脚本
│   └── generation_main.py
├── data/                      # 数据集目录 (包含各个数据集的 parquet 文件和 embeddings)
├── data_preparation/          # 数据预处理脚本
│   ├── add_id_to_mc.py
│   ├── precompute_embeddings.py
│   └── split_dataset.py
├── evaluation/                # 评估代码
│   └── evaluate.py
├── graph_generate/            # 图谱生成引擎
│   ├── extraction_main.py
│   └── retry_main.py
├── graph_match/               # 图匹配模型及训练代码
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── models/                    # 本地模型目录 (权重文件已通过规则忽略)
├── prompts/                   # System / User 提示词模板
│   ├── ablation/
│   ├── cot_gen/
│   ├── graph_gen/
│   └── hallu_detect/
├── utils/                     # 辅助工具包 (模型调用、路径管理、提示词加载等)
├── docker-compose.yml         # Docker 容器配置
├── LICENSE                    # 开源许可证
├── pip_list.txt               # 环境 pip 列表记录
└── requirements.txt           # 项目 Python 依赖项
```
