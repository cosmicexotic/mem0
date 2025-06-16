# Mem0 Evaluation 脚本使用指南

## 📋 概述

本目录包含了Mem0记忆系统的评估脚本，主要用于在LongMemEval数据集上测试Mem0记忆系统的性能。

## 🎯 核心脚本说明

### 1. `reproduce_longmemeval_g.sh` - 图数据库版本实验脚本

**主要用途：**
- 运行基于图数据库的Mem0实验（Mem0+）
- 使用Memgraph图数据库进行记忆存储和检索
- 适用于需要复杂关系推理的场景

**脚本内容分析：**
```bash
dataset_name="longmemeval_oracle_sample_seed_42"
# dataset_name="longmemeval_example"
python evaluation/run_experiments_local_longmemeval.py --technique_type mem0 --dataset_name ${dataset_name} --is_graph

# evaluate the results
python evaluation/evals.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True.json --output_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True_evals.json --dataset longmemeval

# generate the final results
python evaluation/generate_scores.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True_evals.json
```

**关键配置位置：**

#### 数据集名称配置
```bash
# 第1行：修改数据集名称
dataset_name="longmemeval_oracle_sample_seed_42"
# dataset_name="longmemeval_example"
```

**可用的数据集选项：**
- `longmemeval_oracle_sample_seed_42` - 97个样本的采样数据集（推荐用于测试）
- `longmemeval_oracle` - 完整数据集（约15MB）
- `longmemeval_example` - 小示例数据集（25KB）
- `longmemeval_example2` - 另一个示例数据集（72KB）

#### API配置位置

**LLM和图数据库配置位置：`src/memzero/add_local.py` 第74-110行**
```python
config_graph = {
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": "gpt-4o-mini",
                  "api_version": "2025-01-01-preview",
                  "azure_endpoint": "https://123s-mann562s-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                  "api_key": "FjfmeNsmd6aBBbCOWLb4sl8RU0057djGvmGcvzhqYrOkUtifGvd0JQQJ99BEACHYHv6XJ3w3AAAAACOGIlEZ",
              }
        }
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": "text-embedding-3-small",
            "azure_kwargs": {
                "azure_deployment": "text-embedding-3-small",
                "api_version": "2023-05-15",
                "azure_endpoint": "https://123s-mann562s-eastus2.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15",
                "api_key": "FjfmeNsmd6aBBbCOWLb4sl8RU0057djGvmGcvzhqYrOkUtifGvd0JQQJ99BEACHYHv6XJ3w3AAAAACOGIlEZ",
            },
            "embedding_dims": 1536,
        }
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "mem0graph",
        },
    },
}
```

**配置其他API服务：**
如需使用其他API服务（如OpenAI、Anthropic等），请参考Mem0官方文档：
https://docs.mem0.ai/components/llms/models/openai

### 2. `reproduce_longmemeval.sh` - 标准版本实验脚本

**主要用途：**
- 运行标准Mem0实验（不使用图数据库）
- 使用向量数据库进行记忆存储和检索
- 适用于一般记忆检索场景

**与图数据库版本的区别：**
- 不包含 `--is_graph` 参数
- 使用向量数据库而非图数据库
- 结果文件名中 `graph_False` 而非 `graph_True`

### 3. 其他重要脚本

#### `run_experiments_local_longmemeval.py`
- **用途：** 本地运行LongMemEval实验的主要脚本
- **特点：** 支持图数据库和标准模式
- **关键参数：**
  - `--technique_type mem0`: 使用Mem0技术
  - `--is_graph`: 启用图数据库模式
  - `--dataset_name`: 指定数据集名称

#### `run_experiments_longmemeval.py`
- **用途：** 云端运行LongMemEval实验的脚本
- **特点：** 使用Mem0云服务API

## 🚀 运行步骤

### 1. 环境准备

**安装依赖：**
```bash
pip install -r requirements.txt

# 方法2：手动安装核心依赖
pip install mem0 openai python-dotenv tqdm numpy pydantic
pip install langchain-memgraph langchain-neo4j rank-bm25
pip install langchain langchain-openai langgraph langmem
pip install tiktoken jinja2 requests typing-extensions
```

**requirements.txt 说明：**
- **核心依赖：** mem0, openai, python-dotenv, tqdm, numpy, pydantic
- **数据库依赖：** langchain-memgraph, langchain-neo4j, rank-bm25（用于图数据库实验）
- **LLM依赖：** langchain, langchain-openai, langgraph, langmem
- **工具依赖：** tiktoken, jinja2, requests, typing-extensions
- **可选开发工具：** pytest, black

**启动图数据库（如果使用图数据库版本）：**
```bash
# 启动Memgraph
docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph-platform

# 验证连接
docker ps | grep memgraph
```

### 2. 配置API密钥

**修改 `src/memzero/add_local.py` 第74-110行的config_graph字典：**
```python
# 替换以下字段为你的API配置
"api_key": "你的Azure OpenAI API密钥"
"azure_endpoint": "你的Azure OpenAI端点"
```

**配置其他API服务：**
如需使用其他API服务（如OpenAI、Anthropic等），请参考Mem0官方文档：
https://docs.mem0.ai/components/llms/models/openai

### 3. 选择数据集

**修改脚本第1行的数据集名称：**
```bash
# 测试用小数据集
dataset_name="longmemeval_example"

# 正式实验用采样数据集
dataset_name="longmemeval_oracle_sample_seed_42"

# 完整数据集（需要更多时间和资源）
dataset_name="longmemeval_oracle"
```

### 4. 运行实验

#### 图数据库版本（Mem0+）
```bash
bash reproduce_longmemeval_g.sh
```

### 5. 评估结果

**取消注释脚本中的评估行：**
```bash
# 评估实验结果
python evaluation/evals.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True.json --output_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True_evals.json --dataset longmemeval

# 生成最终评分
python evaluation/generate_scores.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_True_evals.json
```

## 📊 结果文件说明

实验完成后，结果文件将保存在 `results/` 目录中：

- `{dataset_name}_mem0_results_top_30_filter_False_graph_True.json` - 原始实验结果
- `{dataset_name}_mem0_results_top_30_filter_False_graph_True_evals.json` - 评估结果

