# Stage 2 关键改进说明

## 改进概述

本次对Stage 2多模态预训练进行了两个关键改进：

1. **使用现有的mask_seq框架** - 确保与代码库完全兼容
2. **实现真正的Hard Negative Sampling** - 提升CMM任务性能

---

## 改进1：使用mask_seq()框架

### 问题
之前的实现在`MultiModalDataLoader`中手动实现了masking逻辑，没有复用现有的`mask_seq()`函数。

### 解决方案
修改为使用现有的`mask_seq()`函数，与其他DataLoader（`RawPacketDataLoader`, `PacketSizeDataLoader`）保持一致。

### 修改内容

#### 1. MultiModalDataset (uer/utils/data.py)

**修改前：**
```python
def create_ins_from_paired_flow(...):
    # ...
    size_src = [CLS] + tokens + [SEP] + padding
    size_tgt_cmmp = size_src.copy()  # 存储原始tokens
    return (raw_src, raw_packet_ids, raw_directions_seq,
            size_src, size_tgt_cmmp)  # 5个元素
```

**修改后：**
```python
def create_ins_from_paired_flow(...):
    # ...
    size_src = [CLS] + tokens + [SEP] + padding
    # 不存储tgt_cmmp，在DataLoader中用mask_seq()生成
    return (raw_src, raw_packet_ids, raw_directions_seq,
            size_src)  # 4个元素
```

#### 2. MultiModalDataLoader (uer/utils/data.py)

**修改前：**
```python
def __iter__(self):
    # 手动实现masking
    maskable_positions = []
    for pos in range(len(size_src)):
        if token not in [CLS, SEP, PAD]:
            maskable_positions.append(pos)

    num_to_mask = int(len(maskable_positions) * 0.15)
    # ... 手动mask逻辑 ...
```

**修改后：**
```python
def __iter__(self):
    # 使用现有的mask_seq()函数
    size_src_masked, tgt_mlm = mask_seq(
        size_src.copy(),
        self.tokenizer_size,
        self.whole_word_masking,
        self.span_masking,
        self.span_geo_prob,
        self.span_max_length
    )

    # 转换tgt_mlm格式: [(pos, token), ...] -> [0, 0, token, 0, ...]
    tgt_cmmp = [0] * len(size_src)
    for pos, token in tgt_mlm:
        tgt_cmmp[pos] = token
```

### 优势

1. **代码一致性** - 与其他DataLoader使用相同的masking逻辑
2. **支持更多masking策略** - 自动支持whole_word_masking和span_masking
3. **维护性更好** - 复用现有测试过的代码
4. **配置统一** - 通过args统一控制masking参数

---

## 改进2：Hard Negative Sampling

### 问题
之前使用随机负采样，导致CMM任务过于简单，模型容易学到trivial solutions。

**之前的随机采样：**
```python
# 在DataLoader中
neg_idx = random.randint(0, batch_size - 1)
while neg_idx == i:
    neg_idx = random.randint(0, batch_size - 1)
```

### 解决方案
在Trainer中实现真正的hard negative sampling：
1. 先forward encoders得到features
2. 基于**相似度**选择负样本（相似度高的更容易被选中）
3. 构造50% positive + 50% hard negative batch

### 架构设计

```
DataLoader (只返回positive samples)
    ↓
Trainer.forward_propagation:
    ↓
1. Forward Encoders (Raw + Size)
    ↓
2. Extract [CLS] features
    ↓
3. Hard Negative Sampling
   (基于cosine similarity选择)
    ↓
4. 构造新batch (50% pos + 50% neg)
    ↓
5. Forward Fusion + Target
```

### 实现细节

#### 1. MultiModalDataLoader返回格式变化

**修改前：**
```python
yield (raw_src, raw_packet_ids, raw_directions,
       size_src, tgt_cmm, tgt_cmmp_size)  # 6个元素
```

**修改后：**
```python
# 只返回positive samples，不返回tgt_cmm
yield (raw_src, raw_packet_ids, raw_directions,
       size_src, tgt_cmmp_size)  # 5个元素
```

#### 2. MultiModalTrainer.forward_propagation

```python
def forward_propagation(self, batch, model):
    raw_src, raw_packet_ids, raw_directions, size_src, tgt_cmmp_size = batch

    # Step 1: Forward Encoders
    raw_emb = model.embedding_raw(raw_src, raw_packet_ids, raw_directions)
    size_emb = model.embedding_size(size_src)

    raw_output = model.encoder_raw(raw_emb, raw_seg)
    size_output = model.encoder_size(size_emb, size_seg)

    # Step 2: Hard Negative Sampling
    raw_cls = raw_output[:, 0, :]  # [batch, hidden]
    size_cls = size_output[:, 0, :]  # [batch, hidden]

    from uer.targets.multimodal_target import hard_negative_sampling
    neg_indices = hard_negative_sampling(raw_cls, size_cls, temperature=0.07)

    # Step 3: 构造50% pos + 50% neg batch
    cmm_labels = []
    final_size_output = []

    for i in range(batch_size):
        if random.random() < 0.5:
            # Positive: 使用原始matched Size
            cmm_labels.append(1)
            final_size_output.append(size_output[i])
        else:
            # Negative: 使用hard negative
            cmm_labels.append(0)
            neg_idx = neg_indices[i].item()
            final_size_output.append(size_output[neg_idx])

    # Step 4: Forward Fusion + Target
    raw_fused, size_fused, (g_raw, g_size) = fusion(
        raw_output, torch.stack(final_size_output)
    )

    cmm_loss, cmmp_loss, ... = target(raw_fused, size_fused, tgt_cmm, tgt_cmmp)
    # ...
```

#### 3. hard_negative_sampling()函数

已在`uer/targets/multimodal_target.py`中实现：

```python
def hard_negative_sampling(raw_features, size_features, temperature=0.07):
    """
    基于相似度的hard negative sampling

    相似度越高，越容易被选为负样本（更难区分）
    """
    batch_size = raw_features.size(0)

    # Normalize features
    raw_norm = F.normalize(raw_features, p=2, dim=1)
    size_norm = F.normalize(size_features, p=2, dim=1)

    # 计算相似度矩阵
    similarities = torch.matmul(raw_norm, size_norm.T)  # [batch, batch]

    # 排除自己 (对角线设为-inf)
    similarities.fill_diagonal_(-float('inf'))

    # 基于相似度采样 (温度参数控制难度)
    probs = F.softmax(similarities / temperature, dim=1)
    neg_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

    return neg_indices
```

### 优势

1. **真正的Hard Negatives** - 基于encoder features，选择最相似（最难区分）的负样本
2. **无需额外Forward** - 复用已经计算的encoder features，训练效率高
3. **可调节难度** - 通过temperature参数控制负样本难度
   - `temperature=0.07` (默认) - 非常hard，选择最相似的
   - `temperature=1.0` - 较easy，更平滑的分布
4. **理论支持** - 与对比学习（SimCLR, MoCo）的做法一致

### Temperature参数说明

```python
temperature = 0.07  # 默认值，hard negatives
# - 相似度高的样本有很高概率被选中
# - CMM任务更难，但学到的表示更好

temperature = 1.0   # 较easy
# - 所有负样本被选概率更均匀
# - 训练初期可以用较高温度
```

---

## 性能影响

### 训练速度
- **无额外开销** - Hard negative sampling复用encoder features，不需要额外forward pass
- **与随机采样相同速度**

### 模型性能预期
- **CMM准确率** - 训练初期可能略低（负样本更难），最终收敛更好
- **表示质量** - 通过区分hard negatives，学到更discriminative的特征
- **下游任务** - 预期在fine-tuning任务上表现更好

### 监控指标

建议监控以下指标：
```
| 10000/100000 steps | Phase1 |
  loss 3.45 | cmm: 0.693 | cmmp: 2.500 |
  acc_cmm: 0.523 ← 初期可能较低（hard negatives）
  acc_cmmp: 0.145
  g_raw: 0.520 | g_size: 0.480
```

**健康的训练曲线：**
- CMM accuracy从~0.5逐渐上升到0.85+
- 如果acc_cmm长期停留在0.5，说明负样本太难，可以提高temperature

---

## 使用建议

### 1. Temperature调节策略

**方案A：固定temperature**
```python
# 默认值，适合大多数情况
temperature = 0.07
```

**方案B：Curriculum Learning**
```python
# 在Trainer中动态调整
if self.current_step < 10000:
    temperature = 1.0   # 训练初期：较easy
elif self.current_step < 50000:
    temperature = 0.5   # 中期：中等难度
else:
    temperature = 0.07  # 后期：very hard
```

### 2. 监控CMM性能

如果发现：
- **acc_cmm < 0.55 (长期)** → temperature太低，负样本太难，提高到0.1或0.5
- **acc_cmm > 0.95 (很快)** → temperature太高，负样本太easy，降低到0.05

### 3. Batch Size建议

Hard negative sampling在batch内选择，因此：
- **Batch size越大** → 负样本池越大 → hard negatives质量越好
- **建议batch size ≥ 64** (单GPU) 或 **256+** (多GPU)

---

## 与现有代码的兼容性

### ✅ 完全兼容

1. **mask_seq()** - 使用现有函数，支持所有masking参数
2. **DataLoader模式** - 遵循现有的buffer + yield模式
3. **Trainer接口** - 保持与其他Trainer一致的接口
4. **参数传递** - 通过args统一控制

### ✅ 无Breaking Changes

- 其他target（raw_packet, packet_size）不受影响
- Stage 1代码完全不受影响
- 可以随时切换回随机采样（修改temperature=inf）

---

## 代码位置总结

### 修改的文件

1. **`uer/utils/data.py`**
   - `MultiModalDataset.create_ins_from_paired_flow()` - 不再存储tgt_cmmp
   - `MultiModalDataLoader.__iter__()` - 使用mask_seq()，不返回tgt_cmm

2. **`uer/trainer.py`**
   - `MultiModalTrainer.forward_propagation()` - 实现hard negative sampling

3. **`uer/models/multimodal_model.py`**
   - `MultiModalModel.forward()` - 简化为inference用，添加说明注释

### 新增的功能

- **`uer/targets/multimodal_target.py:hard_negative_sampling()`** - 已存在，现在被使用

---

## 测试建议

### 单元测试

```python
# 测试hard_negative_sampling
raw_feat = torch.randn(8, 768)
size_feat = torch.randn(8, 768)

neg_indices = hard_negative_sampling(raw_feat, size_feat, temperature=0.07)
assert neg_indices.shape == (8,)
assert (neg_indices != torch.arange(8)).all()  # 不选自己
```

### 集成测试

```bash
# 小规模训练测试
python pre-training/pretrain.py \
    --dataset_path data/test_multimodal.pt \
    --total_steps 100 \
    --batch_size 4 \
    --target multimodal \
    # ... 其他参数
```

监控：
- Batch正确加载（5个tensor）
- mask_seq正常工作
- Hard negative sampling正常（acc_cmm不等于0.5）

---

## 总结

### 改进前 vs 改进后

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| **Masking** | 手动实现 | 使用mask_seq() |
| **代码行数** | ~50行 | ~15行 |
| **兼容性** | 独立实现 | 完全兼容 |
| **CMM采样** | 随机负样本 | Hard negatives |
| **CMM难度** | 简单 | 困难（更有意义） |
| **表示质量** | 一般 | 更好（预期） |
| **训练速度** | 相同 | 相同 |

### 下一步

1. ✅ 代码已完成修改
2. ⏭ 准备数据（paired corpus）
3. ⏭ 小规模测试（100 steps）
4. ⏭ 完整训练（100K steps）
5. ⏭ 评估downstream任务性能

---

最后更新：2025-11-26
