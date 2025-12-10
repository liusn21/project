# ALBEF-style Multi-Modal Pretraining Implementation

## Overview

本文档描述了基于ALBEF (Align Before Fuse) 的多模态预训练实现，用于网络流量分析任务。

### 设计参考
- **ALBEF**: Li et al., "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation"
- **LXMERT**: Tan and Bansal, "LXMERT: Learning Cross-Modality Encoder Representations from Transformers"

### 核心改进
1. **ITC (Image-Text Contrastive)**: 使用对比学习对齐两个模态的表示空间
2. **Momentum Distillation**: 使用EMA更新的momentum encoder + feature queue扩大负样本
3. **6层双向Cross-Attention Fusion**: 深度融合两个模态
4. **Hard Negative Mining**: ITM任务使用ITC相似度进行困难负样本挖掘

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 2 Multi-Modal Pretraining                  │
└─────────────────────────────────────────────────────────────────────┘

[Raw Packet Data]                              [Packet Size Data]
       ↓                                              ↓
┌──────────────────┐                         ┌──────────────────┐
│  Raw Embedding   │                         │  Size Embedding  │
│ (token+pos+pkt+  │                         │  (token+pos)     │
│    direction)    │                         │                  │
└────────┬─────────┘                         └────────┬─────────┘
         ↓                                            ↓
┌──────────────────┐                         ┌──────────────────┐
│   Raw Encoder    │ (from Stage 1)          │  Size Encoder    │
│  (12-layer Xfmr) │                         │  (12-layer Xfmr) │
└────────┬─────────┘                         └────────┬─────────┘
         │                                            │
         ↓                                            ↓
    [raw_output]                                [size_output]
    [B, 512, 768]                               [B, 256, 768]
         │                                            │
         ├────────── ITC Loss ────────────────────────┤
         │         (对比学习对齐)                       │
         │         + Momentum Queue                   │
         │                                            │
         └────────────────┬───────────────────────────┘
                          ↓
         ┌────────────────────────────────────────────┐
         │       Bidirectional Fusion Encoder         │
         │              (6 layers)                    │
         │                                            │
         │  Each layer:                               │
         │  ├─ Self-Attention (raw)                   │
         │  ├─ Self-Attention (size)                  │
         │  ├─ Cross-Attention (Q=raw, KV=size)       │
         │  ├─ Cross-Attention (Q=size, KV=raw)       │
         │  ├─ FFN (raw)                              │
         │  └─ FFN (size)                             │
         └────────────────┬───────────────────────────┘
                          ↓
         ┌────────────────┴────────────────────────────┐
         ↓                                             ↓
   [raw_fused]                                   [size_fused]
         │                                             │
         └─────────────────┬───────────────────────────┘
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
      ITM Loss        MLM_raw Loss      MLM_size Loss
 (Hard Neg Mining)

Total Loss = λ_itc * L_ITC + λ_itm * L_ITM + λ_mlm * (L_MLM_raw + L_MLM_size)
```

---

## Modified Files

### 1. `uer/layers/multimodal_fusion.py` (完全重写)

**改动说明**: 移除Gate机制，实现6层双向Cross-Attention

```python
# 新增类
class BidirectionalFusionLayer(nn.Module):
    """单层双向Cross-Attention Fusion"""
    # 包含: Self-Attn(raw) + Self-Attn(size) + Cross-Attn双向 + FFN双向

class MultiModalFusionEncoder(nn.Module):
    """6层Fusion Encoder"""
    # 包含: 6个BidirectionalFusionLayer + mask生成
```

### 2. `uer/targets/multimodal_target.py` (完全重写)

**改动说明**: 实现ALBEF的三个任务

```python
class MultiModalTarget(nn.Module):
    # forward_itc(): ITC对比学习，使用momentum queue
    # forward_itm(): ITM匹配任务，使用hard negative mining
    # forward_mlm_raw(): Raw模态MLM
    # forward_mlm_size(): Size模态MLM
```

### 3. `uer/models/multimodal_model.py` (完全重写)

**改动说明**: 实现带Momentum Encoder和Queue的模型

```python
class MultiModalModel(nn.Module):
    # 包含:
    # - embedding_raw, encoder_raw (主encoder)
    # - embedding_size, encoder_size (主encoder)
    # - embedding_raw_m, encoder_raw_m (momentum encoder, requires_grad=False)
    # - embedding_size_m, encoder_size_m (momentum encoder, requires_grad=False)
    # - fusion (6层Fusion)
    # - target (ITC+ITM+MLM)
    # - raw_queue, size_queue (feature queues)

    # _momentum_update(): EMA更新momentum encoder
    # _dequeue_and_enqueue(): 更新feature queue
```

### 4. `uer/trainer.py`

**改动说明**: 更新MultiModalTrainer

```python
class MultiModalTrainer(Trainer):
    # 改动:
    # - 移除phase1/phase2逻辑
    # - 新增ITC/ITM/MLM loss跟踪
    # - 简化forward_propagation调用model.forward()
```

### 5. `uer/model_builder.py` (部分重写)

**改动说明**: 更新multimodal模型构建逻辑

```python
# 改动:
# - 创建MultiModalFusionEncoder
# - 创建MultiModalTarget (新签名)
# - 加载Stage 1预训练encoder到主encoder和momentum encoder
```

### 6. `pre-training/pretrain.py`

**改动说明**: 更新命令行参数

```python
# 移除参数:
# --phase1, --phase2
# --balance_loss_alpha, --cmmp_raw_weight, --cmmp_size_weight
# --gate_temperature, --cmm_temperature

# 新增参数:
--num_fusion_layers  # Fusion层数，默认6
--queue_size         # ITC queue大小，默认4096
--momentum           # EMA系数，默认0.995
--lambda_itc         # ITC loss权重，默认1.0
--lambda_itm         # ITM loss权重，默认1.0
--lambda_mlm         # MLM loss权重，默认1.0
--itc_temperature    # ITC温度，默认0.07
--itm_temperature    # ITM温度，默认0.07
```

---

## Training Commands

### Prerequisites

确保已完成Stage 1预训练:
- Raw Packet encoder: `models/raw_packet_encoder.bin`
- Packet Size encoder: `models/size_encoder.bin`

### Stage 2: Multi-Modal Pretraining (ALBEF-style)

```bash
python pre-training/pretrain.py \
    --dataset_path data/multimodal_dataset.pt \
    --vocab_path_raw vocab/raw_vocab.txt \
    --vocab_path_size vocab/size_vocab.txt \
    --pretrained_raw_path models/raw_packet_encoder.bin \
    --pretrained_size_path models/size_encoder.bin \
    --output_model_path models/multimodal_albef \
    --config_path models/bert/base_config.json \
    --target multimodal \
    --encoder transformer \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --warmup 0.1 \
    --scheduler linear \
    --num_fusion_layers 6 \
    --queue_size 4096 \
    --momentum 0.995 \
    --lambda_itc 1.0 \
    --lambda_itm 1.0 \
    --lambda_mlm 1.0 \
    --itc_temperature 0.07 \
    --itm_temperature 0.07 \
    --encoder_lr_ratio 0.1 \
    --world_size 1 \
    --gpu_ranks 0
```

### Key Hyperparameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_fusion_layers` | 6 | Fusion层数 |
| `queue_size` | 4096 | ITC feature queue大小 |
| `momentum` | 0.995 | Momentum encoder EMA系数 |
| `lambda_itc` | 1.0 | ITC loss权重 |
| `lambda_itm` | 1.0 | ITM loss权重 |
| `lambda_mlm` | 1.0 | MLM loss权重 |
| `itc_temperature` | 0.07 | ITC对比学习温度 |
| `itm_temperature` | 0.07 | ITM hard negative采样温度 |
| `encoder_lr_ratio` | 0.1 | Encoder学习率 = base_lr * ratio |

---

## Training Output

训练过程中会输出以下指标:

```
| steps/total | time | loss | itc | itm | mlm_r | mlm_s | acc_itm | acc_mlm_r | acc_mlm_s |
```

- `loss`: 总loss
- `itc`: ITC对比学习loss
- `itm`: ITM匹配任务loss
- `mlm_r`: Raw模态MLM loss
- `mlm_s`: Size模态MLM loss
- `acc_itm`: ITM准确率
- `acc_mlm_r`: Raw MLM准确率
- `acc_mlm_s`: Size MLM准确率

---

## Key Implementation Details

### 1. Momentum Distillation

```python
# EMA更新 (每个训练step)
param_m.data = param_m.data * momentum + param.data * (1 - momentum)

# momentum encoder不参与梯度计算
for param in momentum_encoder.parameters():
    param.requires_grad = False
```

### 2. Feature Queue

```python
# Queue用于扩大ITC负样本数量
# 每个batch后更新queue
raw_queue = concat(raw_queue, raw_feat_m)  # FIFO
size_queue = concat(size_queue, size_feat_m)
```

### 3. Hard Negative Mining (ITM)

```python
# 使用ITC相似度矩阵采样hard negatives
sim_r2s.fill_diagonal_(-inf)  # mask正样本
weights = softmax(sim_r2s / temperature)
neg_idx = multinomial(weights, 1)  # 按相似度采样
```

### 4. Differential Learning Rate

```python
# Encoder使用较小学习率
encoder_lr = base_lr * encoder_lr_ratio  # 0.1x
# Fusion和Target使用正常学习率
fusion_lr = base_lr
```

---

## Comparison with Original Design

| 方面 | 原设计 | ALBEF-style |
|------|--------|-------------|
| Fusion | 1层 + Gate | 6层双向Cross-Attention |
| 对齐方式 | 无 (仅CMM in fusion后) | ITC (fusion前对比学习) |
| 负样本 | batch内 | batch + queue (扩大负样本) |
| ITM | 简单负样本 | Hard negative mining |
| Phase训练 | Phase1冻结 + Phase2全参 | 统一全参 + differential LR |
| MLM | CMMP (fusion后) | 同样fusion后，但有ITC对齐支持 |

---

## Notes

1. **数据格式**: 确保MultiModalDataset返回的batch格式为:
   `(raw_src, raw_packet_ids, raw_directions, size_src, tgt_mlm_raw, tgt_mlm_size)`

2. **显存占用**: 由于有momentum encoder (双倍encoder参数) 和 queue，显存占用较大。
   - 建议batch_size从16开始尝试
   - queue_size可以根据显存调整

3. **训练稳定性**:
   - ITC在初期可能不稳定，建议warmup
   - 如果loss震荡，可以尝试降低学习率

4. **预训练权重加载**:
   - Stage 1预训练的encoder会同时加载到主encoder和momentum encoder
   - 确保Stage 1模型保存格式包含`embedding.`和`encoder.`前缀
