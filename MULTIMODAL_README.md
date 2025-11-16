# Multi-Modal Traffic Analysis Pre-training

基于 TrafficFormer 的多模态流量分析预训练实现，结合 PTU 的多模态思想和 SimSiam 对比学习。

## 概述

本实现扩展了 TrafficFormer，添加了以下功能：

1. **多模态 Tokenizer**：
   - Raw packet tokens (hex bigram): 65,536 词汇量
   - Temporal tokens (IAT): 1,000 词汇量
   - Size+Direction tokens: 3,001 词汇量

2. **预训练方法**：
   - SimSiam 对比学习（主要目标，待实现数据增强）
   - MBM (Masked Burst Modeling) 辅助任务

3. **完整的训练管道**：
   - PCAP → JSON 特征提取
   - Multi-modal Dataset/DataLoader
   - Multi-modal Embedding
   - SimSiam+MBM Target
   - 自定义 Trainer

## 架构设计

### 1. Token 结构

每个数据包的 token 序列：
```
[packet_tokens(63)] + [temporal_tokens(1-10)] + [size_token(1)]
```

- **Packet tokens**: 64 字节的 hex bigram (63 tokens)
- **Temporal tokens**: 前 min(n, 10) 个包的 IAT，编码为 sigmoid(log10(IAT))
- **Size token**: size * direction + 1500，范围 [0, 3000]

### 2. 词汇表设计

```python
packet_vocab_size = 65536    # 0x0000 - 0xFFFF
temporal_vocab_size = 1000   # IAT tokens
size_vocab_size = 3001       # Size+Direction tokens
total = 69,537 tokens
```

### 3. Embedding 架构

```python
class MultiModalEmbedding:
    - packet_embedding: (65536, emb_size)
    - temporal_embedding: (1000, emb_size)
    - size_embedding: (3001, emb_size)
    - type_embedding: (3, emb_size)  # 区分三种模态
    - position_embedding: (seq_len, emb_size)
```

## 使用方法

### Step 1: PCAP → JSON 特征提取

```bash
python data_generation/multimodal_data_gen.py \
    --pcap_dir /path/to/pcap/files \
    --output_json data/multimodal_flows.json \
    --bytes_per_packet 64 \
    --min_packets 5
```

**输出格式** (JSON):
```json
{
  "num_flows": 1000,
  "flows": [
    {
      "flow_id": "flow_12345",
      "protocol": 6,
      "num_packets": 10,
      "raw_tokens": [["4500", "5000", ...], ...],
      "temporal_tokens": [[123, 456], [789], ...],
      "size_tokens": [1800, 2100, ...],
      "directions": [1, -1, 1, ...]
    }
  ]
}
```

### Step 2: 预训练

```bash
python pretrain_multimodal.py \
    --json_path data/multimodal_flows.json \
    --output_model_path models/multimodal_pretrain.bin \
    --config_path models/bert/base_config.json \
    --embedding multimodal \
    --encoder transformer \
    --target simsiam_mbm \
    --seq_length 512 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --mbm_weight 0.1 \
    --projection_dim 2048 \
    --gpu_ranks 0
```

**训练输出示例**:
```
| 100/100000 steps | 1234.56 tokens/s | loss 5.42 | loss_simsiam 0.0000 | loss_mbm 5.42 | acc_mbm: 0.123
```

### Step 3: 微调（待实现）

下游任务微调代码待开发，将支持：
- 类别平衡采样
- 监督对比损失

## 代码结构

```
project/
├── data_generation/
│   └── multimodal_data_gen.py          # PCAP→JSON 特征提取
├── uer/
│   ├── layers/
│   │   ├── embeddings.py               # + MultiModalEmbedding
│   │   └── __init__.py                 # 注册 multimodal embedding
│   ├── models/
│   │   └── model.py                    # + MultiModalModel
│   ├── targets/
│   │   ├── simsiam_mbm_target.py       # SimSiam + MBM
│   │   └── __init__.py                 # 注册 simsiam_mbm target
│   ├── utils/
│   │   ├── data.py                     # + MultiModalDataset/DataLoader
│   │   └── __init__.py                 # 注册 multimodal dataset
│   ├── model_builder.py                # 支持 MultiModalModel
│   └── trainer.py                      # + MultiModalTrainer
└── pretrain_multimodal.py              # 训练脚本
```

## 新增组件详解

### 1. MultiModalDataset

**功能**:
- 从 JSON 读取流量特征
- 构建 token 序列（raw + temporal + size）
- 完整包截断策略
- BERT-style 15% masking

**关键方法**:
```python
def _create_instances_from_flow(flow):
    # 构建: [raw(63)] + [temporal(1-10)] + [size(1)] per packet
    # 返回: (tokens, token_types, positions)

def _truncate_to_complete_packets(tokens, types, positions):
    # 按完整包截断到 seq_length

def _apply_masking(src, token_types):
    # 15% masking, 分别对三种 token 类型
```

### 2. MultiModalEmbedding

**功能**:
- 三个独立的 embedding 层
- Type embedding 区分模态
- Position embedding

**前向传播**:
```python
def forward(src, token_types, positions):
    # 根据 token_types 路由到不同的 embedding
    # token_emb + type_emb + pos_emb
```

### 3. SimSiamMBMTarget

**功能**:
- SimSiam 对比学习（待数据增强）
- MBM 多头输出（针对三种 token 类型）

**损失函数**:
```python
loss_total = loss_simsiam + mbm_weight * loss_mbm
```

**当前状态**:
- MBM: ✅ 已实现
- SimSiam: ⏸️ 待数据增强实现

### 4. MultiModalTrainer

**功能**:
- 处理 (src, tgt, token_types, positions) 批次
- 统计 SimSiam 和 MBM 损失
- 报告训练进度

## 待实现功能

### 1. 数据增强 (高优先级)

**时域增强**:
- 时序偏移 (jitter)
- 模拟重传
- 包延迟

**频域增强**:
- FFT 变换
- 添加噪声
- 频谱混合

**实现位置**: `data_generation/augmentation.py`

### 2. SimSiam 对比学习

**依赖**: 数据增强模块

**实现步骤**:
1. 实现增强函数
2. 修改 MultiModalModel 创建两个 view
3. 激活 SimSiamMBMTarget 的对比学习分支

### 3. 下游任务微调

**任务**:
- 流量分类
- 应用识别
- 异常检测

**需要实现**:
- Fine-tuning 数据集
- 类别平衡采样器
- 监督对比损失

## 超参数建议

### 预训练
```python
seq_length = 512         # 可容纳约 7 个完整包
batch_size = 32          # 根据 GPU 内存调整
learning_rate = 2e-4     # BERT-style
total_steps = 100000     # 根据数据集大小调整
warmup = 0.1             # 10% warmup
dropout = 0.1

# Multi-modal specific
mbm_weight = 0.1         # MBM 辅助任务权重
projection_dim = 2048    # SimSiam projection head
mask_ratio = 0.15        # BERT-style masking
```

### 模型配置 (base_config.json)
```json
{
  "hidden_size": 768,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "intermediate_size": 3072,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "max_position_embeddings": 512
}
```

## 性能估算

### 内存占用
- Embedding 层: ~214 MB
- Transformer (12层): ~300 MB
- 批次数据 (batch=32): ~50 MB
- **总计**: ~600 MB (仅模型)

### 训练速度（估算）
- V100 GPU: ~1000-1500 tokens/s
- 100K steps: ~10-15 小时（取决于硬件）

## 调试和验证

### 1. 检查数据格式
```python
import json
with open('data/multimodal_flows.json') as f:
    data = json.load(f)
print(f"Flows: {data['num_flows']}")
print(f"Sample flow: {data['flows'][0]}")
```

### 2. 测试 Dataset
```python
from uer.utils.data import MultiModalDataset

dataset = MultiModalDataset(args, 'data/multimodal_flows.json')
src, tgt, types, pos = dataset[0]
print(f"Token shape: {src.shape}")
print(f"Types: {types}")
```

### 3. 小规模训练测试
```bash
python pretrain_multimodal.py \
    --json_path data/test.json \
    --output_model_path models/test.bin \
    --batch_size 4 \
    --total_steps 100 \
    --seq_length 512
```

## 问题排查

### 常见错误

**1. Dimension mismatch in embedding**
- 检查 token_types 是否在 [0, 1, 2] 范围内
- 检查 src tokens 是否在对应词汇表范围内

**2. OOM (Out of Memory)**
- 减小 batch_size
- 减小 seq_length
- 使用 gradient_accumulation

**3. NaN loss**
- 检查 temporal token 计算（避免 log(0)）
- 降低 learning_rate
- 检查数据是否有异常值

## 性能优化建议

1. **数据预处理**: 预先生成 .pt 文件而不是 JSON (更快加载)
2. **Mixed Precision**: 使用 fp16 训练（加速 2x）
3. **分布式训练**: 多 GPU 并行
4. **Gradient Checkpointing**: 减少内存占用

## 引用

本实现基于以下论文和项目：

- **TrafficFormer**: Encrypted network traffic classification using transformer
- **PTU**: Pre-trained Traffic Understanding with temporal tokens
- **SimSiam**: Simple Siamese Representation Learning

## 联系和贡献

如有问题或改进建议，请提交 issue 或 PR。
