# Stage 2 多模态预训练架构 v2

## 📋 架构变更说明

### 关键问题修复

**原架构问题**：
- CMM任务在Fusion**之前**计算（使用encoder输出）
- Phase 1冻结encoder时，CMM loss无法更新任何参数 ❌
- 梯度流被阻断，导致Phase 1训练无效

**新架构解决方案**：
- **CMM和CMMP都在Fusion之后**
- Phase 1冻结encoder时，所有loss都能训练fusion/target ✅
- CMM使用fused features，质量更高 ✅
- 动态负样本挖掘，基于fused features ✅

---

## 🔄 完整训练流程

```
┌─────────────────────────────────────────────────────────┐
│           DataLoader (只返回正样本对)                      │
│  Batch: (raw_src, raw_packet_ids, raw_directions,      │
│          size_src, tgt_mlm_size)                        │
│  所有样本都是匹配的模态对: (Raw_i, Size_i)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ batch_size = 16 (举例)
                     ↓
        ┌────────────────────────────┐
        │  Step 1: Encoder Forward   │
        │  Phase 1: FROZEN ❄️        │
        │  Phase 2: Trainable 🔥     │
        └────────────┬───────────────┘
                     ↓
        raw_output [16, 512, 768]
        size_output [16, 256, 768]
                     │
                     ↓
        ┌────────────────────────────┐
        │  Step 2: Fusion            │
        │  Phase 1/2: Trainable 🔥   │
        │  - Cross-Attention          │
        │  - Gating (sample-level)    │
        │  - 传递seg mask             │
        └────────────┬───────────────┘
                     ↓
        raw_fused [16, 512, 768]
        size_fused [16, 256, 768]
        g_raw, g_size [16, 1]
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
┌──────────────────┐    ┌──────────────────┐
│  Step 3: CMMP    │    │  Step 4: CMM     │
│  (MLM任务)       │    │  (ITM任务)        │
│  Fusion后        │    │  Fusion后         │
└──────────────────┘    └──────────────────┘
```

---

## 📌 Step 3: CMMP任务（跨模态掩码预测）

**任务类型**：Masked Language Modeling (MLM)

**输入**：
- `size_fused` [batch, 256, 768] - Fused Size序列
- `tgt_mlm_size` [batch, 256] - 被mask的Size token targets

**样本使用**：
- ✅ **所有正样本** (batch内所有样本)
- 因为要预测Size_i的masked tokens，必须用匹配的Raw_i

**实现**：
```python
cmmp_loss, cmmp_correct, cmmp_denominator = target.forward_cmmp_only(
    size_fused, tgt_cmmp_size
)
```

**梯度流向**：
- Phase 1: CMMP loss → Target head → Fusion ✅
- Phase 2: CMMP loss → Target head → Fusion → Encoders ✅

---

## 📌 Step 4: CMM任务（跨模态匹配 - 标准ITM二分类）

**任务类型**：Image-Text Matching (ITM) / Binary Classification

**输入**：
- `raw_fused[:, 0, :]` [batch, 768] - Fused Raw [CLS]
- `size_fused[:, 0, :]` [batch, 768] - Fused Size [CLS]

**实现方式：Element-wise Product + MLP**

### 4.1 Hard Negative Mining（困难负样本挖掘）

```python
# 在batch内，基于fused [CLS] features
raw_cls = raw_fused[:, 0, :]  # [16, 768]
size_cls = size_fused[:, 0, :]  # [16, 768]

# 计算相似度矩阵（用于采样，不用于loss）
with torch.no_grad():
    raw_norm = F.normalize(raw_cls, p=2, dim=1)
    size_norm = F.normalize(size_cls, p=2, dim=1)
    similarities = torch.matmul(raw_norm, size_norm.T)  # [16, 16]
    similarities.fill_diagonal_(-inf)  # 排除自己

    # Hard negative sampling (相似度高的更容易被选中)
    probs = softmax(similarities / temperature)
    neg_indices = multinomial(probs)  # [16]
```

### 4.2 构建训练样本（50% pos + 50% neg）

```python
对于每个样本i（i=0到15）：
  随机数 = torch.rand()

  if 随机数 < 0.5:
    # Positive sample
    raw_features[i] = raw_cls[i]
    size_features[i] = size_cls[i]
    labels[i] = 1.0
  else:
    # Negative sample (hard negative)
    j = neg_indices[i]
    raw_features[i] = raw_cls[i]
    size_features[i] = size_cls[j]  # 不匹配的size
    labels[i] = 0.0
```

### 4.3 Element-wise Product + MLP

```python
# 点乘捕捉交互特征
interaction = raw_features * size_features  # [16, 768]

# MLP分类头
logits = ITM_head(interaction)  # 768 -> 384 -> 1

# ITM_head结构:
# Linear(768, 384) -> ReLU -> Dropout -> Linear(384, 1)
```

### 4.4 Binary Classification Loss

```python
# Binary cross-entropy loss
cmm_loss = BCEWithLogits(logits, labels)

# Accuracy
pred = (sigmoid(logits) > 0.5)
cmm_correct = (pred == labels).sum()
```

**实现**：
```python
cmm_loss, cmm_correct = target.forward_cmm_itm(
    raw_fused, size_fused, temperature=self.cmm_temperature
)
```

**梯度流向**：
- Phase 1: CMM loss → Fusion (通过[CLS]) ✅
- Phase 2: CMM loss → Fusion → Encoders ✅

---

## 🔄 两阶段训练

### Phase 1 (0 - 70K steps): Freeze Encoders

**冻结组件**：
- ❄️ Encoder_Raw
- ❄️ Encoder_Size

**可训练组件**：
- 🔥 Fusion module
- 🔥 Target module (CMM + CMMP heads)

**梯度流**：
```
CMM loss → Fusion ✅
CMMP loss → Fusion ✅
Balance loss → Fusion ✅

所有loss都可以更新Fusion和Target ✅
```

### Phase 2 (70K - 100K steps): Joint Training

**全部可训练**：
- 🔥 Encoder_Raw
- 🔥 Encoder_Size
- 🔥 Fusion module
- 🔥 Target module

**梯度流**：
```
所有loss → Target/Fusion → Encoders ✅
端到端微调整个模型 ✅
```

---

## 📊 v2 vs v1 对比

| 方面 | v1 (CMM在fusion前) | v2 (CMM在fusion后) |
|-----|-------------------|-------------------|
| **Phase 1梯度流** | ❌ CMM loss无法更新任何参数 | ✅ 所有loss都更新fusion/target |
| **CMM特征质量** | ⚠️ Encoder features | ✅ Fused features (更强) |
| **CMMP特征质量** | ✅ Fused features | ✅ Fused features |
| **负样本构建** | ⚠️ 基于encoder features | ✅ 基于fused features |
| **负样本多样性** | ⚠️ 每个epoch固定 | ✅ 动态（每次不同） |
| **任务一致性** | ⚠️ CMM用encoder，CMMP用fusion | ✅ 都用fusion |
| **实现复杂度** | ⚠️ 需要DataLoader返回负样本 | ✅ Batch内动态构建 |

---

## 🎯 关键创新

1. **标准ITM二分类 + 困难负样本挖掘**
   - Element-wise product捕捉交互特征
   - MLP分类头（768→384→1）
   - 困难负样本挖掘（基于相似度）
   - 适合小batch（16即可）

2. **动态Hard Negative Mining**
   - 在batch内基于fused features挖掘
   - 每个iteration都不同
   - 难度自适应增长
   - 相似度高的负样本更容易被选中

3. **统一特征空间**
   - CMM和CMMP都使用fused features
   - 特征质量更高、更一致

4. **梯度流优化**
   - Phase 1所有loss都能训练fusion/target
   - 避免了梯度阻断问题

---

## 🆚 为什么选择ITM而不是CLIP？

| 方面 | CLIP对比学习 | ITM二分类 | 我们的选择 |
|-----|------------|----------|----------|
| **Loss类型** | InfoNCE (多分类) | Binary Cross-Entropy | ✅ ITM |
| **Batch要求** | 需要大batch (>256) | 小batch即可 (16) | ✅ ITM (batch=16) |
| **负样本数** | batch_size - 1 | 自定义（50%） | ✅ ITM (更灵活) |
| **困难负样本** | 无法单独控制 | 可以hard negative mining | ✅ ITM |
| **计算复杂度** | O(batch²) | O(batch) | ✅ ITM (更高效) |
| **适用场景** | 大规模对比学习 | 小batch匹配任务 | ✅ ITM |

**选择ITM的原因**：
1. ✅ **Batch size限制**：我们的batch=16，不适合CLIP（需要几百）
2. ✅ **困难负样本有效**：二分类任务可以利用hard negative mining
3. ✅ **计算高效**：只计算选中的样本，不需要batch内所有配对
4. ✅ **语义清晰**：明确的匹配/不匹配二分类
5. ✅ **实现简单**：Element-wise product + MLP，易于理解和调试

---

## 📝 代码修改清单

### 1. uer/targets/multimodal_target.py

**新增ITM分类头**：
```python
# CMM: ITM二分类头（element-wise product + MLP）
self.itm_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384
    nn.ReLU(),
    nn.Dropout(args.dropout),
    nn.Linear(hidden_size // 2, 1)  # 384 -> 1
)
self.itm_criterion = nn.BCEWithLogitsLoss()
```

**核心方法**：
```python
def forward_cmm_itm(self, raw_fused, size_fused, temperature=0.07):
    """
    标准ITM二分类任务：
    1. 困难负样本挖掘（基于相似度）
    2. 构建50% pos + 50% neg
    3. Element-wise product
    4. MLP分类
    """
    # Step 1: Hard negative mining
    with torch.no_grad():
        similarities = cosine_sim(raw_cls, size_cls)
        neg_indices = multinomial(softmax(similarities / temperature))

    # Step 2: 构建训练样本
    for i in range(batch_size):
        if random() < 0.5:
            样本 = (raw_cls[i], size_cls[i], label=1)
        else:
            样本 = (raw_cls[i], size_cls[neg_indices[i]], label=0)

    # Step 3-4: Element-wise product + MLP
    interaction = raw_features * size_features
    logits = self.itm_head(interaction)
    loss = BCEWithLogits(logits, labels)
```

**保留方法**：
- `MultiModalTarget.forward_cmmp_only()` - CMMP任务（已经在fusion后）

### 2. uer/trainer.py

**修改**：
- `MultiModalTrainer.forward_propagation()` - 调整任务顺序
  - Step 1: Encoder
  - Step 2: Fusion
  - Step 3: CMMP (fusion后)
  - Step 4: CMM (fusion后) ← 新位置

### 3. uer/layers/multimodal_fusion.py

**保持不变** - 已经支持seg mask传递

### 4. uer/models/multimodal_model.py

**保持不变** - forward()主要用于inference

---

## ⚡ 使用示例

```bash
python pretrain.py \
    --dataset_path data/multimodal_dataset.pt \
    --vocab_path_raw models/vocab_raw.txt \
    --vocab_path_size models/vocab_size.txt \
    --pretrained_raw_path models/stage1_raw.bin \
    --pretrained_size_path models/stage1_size.bin \
    --output_model_path models/stage2_model.bin \
    --config_path models/multimodal_config.json \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 16 \
    --target multimodal \
    --freeze_encoders \
    --phase1_steps 70000 \
    --balance_loss_alpha 0.1 \
    --gate_temperature 0.5 \
    --cmm_temperature 0.07
```

---

## ✅ 验证清单

- [x] Phase 1梯度流正确（所有loss都能训练fusion/target）
- [x] CMM使用fused features
- [x] CMMP使用fused features
- [x] 动态负样本构建（batch内hard negative mining）
- [x] Padding mask正确传递
- [x] 统计信息正确计算
- [x] 两阶段训练逻辑正确

---

## 📈 期望效果

1. **Phase 1训练有效** - 所有loss都能更新参数
2. **特征质量提升** - CMM和CMMP都用fused features
3. **负样本质量提升** - 基于fused features的hard negatives
4. **训练稳定性** - 动态负样本，避免过拟合
5. **模型性能提升** - 端到端优化，特征更一致

---

生成时间：2025-12-03
架构版本：v2
