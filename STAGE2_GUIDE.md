# Stage 2: Multi-Modal Pretraining Guide

## Overview

Stage 2 implements multi-modal pretraining that fuses the two pretrained encoders from Stage 1 (Raw Packet + Packet Size) using cross-attention and gating mechanisms.

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Raw Packet     │         │  Packet Size    │
│  Encoder        │         │  Encoder        │
│  (Pretrained)   │         │  (Pretrained)   │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │  [batch, 512, hidden]     │  [batch, 256, hidden]
         │                           │
         └─────────┬─────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Fusion Module    │
         │  - Cross-Attention│
         │  - Gating (MOE)   │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Target Module    │
         │  - CMM Task       │
         │  - CMMP Task      │
         │  - Balance Loss   │
         └───────────────────┘
```

### Pretraining Tasks

1. **CMM (Cross-Modal Matching)**
   - Binary classification: Are Raw and Size from the same flow?
   - 50% positive samples, 50% negative samples
   - Negative sampling: Random (hard negative sampling can be added later)
   - Loss: Binary Cross-Entropy on cosine similarity of [CLS] tokens

2. **CMMP (Cross-Modal Masked Prediction)**
   - Predict masked Size tokens from Raw features
   - Single direction: Raw → Size (based on protocol structure)
   - Masking: 15% of Size tokens (80% [MASK], 10% random, 10% unchanged)
   - Loss: Cross-Entropy on masked positions

3. **Balance Loss**
   - Prevent modality imbalance in gating mechanism
   - Encourage equal usage of Raw and Size modalities (target: 0.5 each)
   - Weight: λ = 0.1

### Two-Phase Training

**Phase 1 (Steps 1-70,000)**: Frozen Encoders
- Freeze: `encoder_raw`, `encoder_size`, `embedding_raw`, `embedding_size`
- Train: `fusion` module only
- Learning rate: 2e-5
- Goal: Warm up fusion module without polluting pretrained encoders

**Phase 2 (Steps 70,001-100,000)**: Joint Training
- Unfreeze: ALL parameters
- Differential learning rates:
  - Encoders: 5e-6 (lower to preserve pretrained knowledge)
  - Fusion: 2e-5 (normal rate)
- Goal: Fine-tune entire model end-to-end

---

## What Was Modified

### New Files Created

1. **`uer/layers/multimodal_fusion.py`**
   - `GatedMultiModalFusion`: Cross-attention + gating mechanism
   - `compute_balance_loss()`: Balance loss computation
   - MM4flow-style concatenated query cross-attention

2. **`uer/targets/multimodal_target.py`**
   - `MultiModalTarget`: CMM + CMMP tasks
   - `hard_negative_sampling()`: Hard negative sampling for CMM

3. **`uer/models/multimodal_model.py`**
   - `MultiModalModel`: Integrates two encoders + fusion + target
   - `freeze_encoders()` / `unfreeze_encoders()`: Phase control

4. **`uer/utils/data.py`** (additions)
   - `MultiModalDataset`: Loads paired (Raw, Size) flows
   - `MultiModalDataLoader`: Implements CMM sampling + CMMP masking

5. **`uer/trainer.py`** (additions)
   - `MultiModalTrainer`: Two-phase training logic
   - Tracks: CMM loss/acc, CMMP loss/acc, balance loss, gate weights

### Modified Files

1. **`uer/model_builder.py`**
   - Added multimodal model building logic
   - Loads pretrained encoders from `--pretrained_raw_path` and `--pretrained_size_path`

2. **`uer/utils/__init__.py`**
   - Registered `MultiModalDataset` and `MultiModalDataLoader`

3. **`uer/targets/__init__.py`**
   - Registered `MultiModalTarget`

---

## Data Preparation

Stage 2 requires **paired** Raw Packet + Packet Size data in the **same order**.

### Expected Data Format

You need TWO corpus files:

1. **Raw Packet Corpus** (`corpus_raw.txt`)
   ```
   ||
   6
   1 4500 0006 0683 3c52 5297 ...
   -1 4500 0000 003c 3c52 ...
   ||
   17
   1 ...
   ```

2. **Packet Size Corpus** (`corpus_size.txt`)
   ```
   ||
   6
   1672 2185 953 ...
   ||
   17
   1300 1400 ...
   ```

**CRITICAL**: Both files must have:
- Same number of flows
- Same flow order
- Same protocol information

### Building Multi-Modal Dataset

```bash
python data_generation/build_multimodal_dataset.py \
    --corpus_path_raw data/corpus_raw.txt \
    --corpus_path_size data/corpus_size.txt \
    --vocab_path_raw models/vocab_raw.txt \
    --vocab_path_size models/vocab_size.txt \
    --dataset_path data/multimodal_dataset.pt \
    --seq_length 512 \
    --workers_num 8 \
    --dynamic_masking \
    --docs_buffer_size 100000 \
    --dup_factor 1
```

**Note**: This script needs to be created if it doesn't exist. You can also build datasets on-the-fly during training.

---

## Training Stage 2

### Phase 1: Frozen Encoders (0-70K steps)

```bash
python pre-training/pretrain.py \
    --dataset_path data/multimodal_dataset.pt \
    --vocab_path_raw models/vocab_raw.txt \
    --vocab_path_size models/vocab_size.txt \
    --pretrained_raw_path models/raw_encoder_30000.bin \
    --pretrained_size_path models/size_encoder_30000.bin \
    --output_model_path models/multimodal_stage2 \
    --config_path models/bert/base_config.json \
    --total_steps 70000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --warmup 0.1 \
    --target multimodal \
    --freeze_encoders \
    --phase1_steps 70000 \
    --balance_loss_alpha 0.1 \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --dist_train \
    --backend nccl
```

### Phase 2: Joint Training (70K-100K steps)

After Phase 1 completes, start Phase 2 from the Phase 1 checkpoint:

```bash
python pre-training/pretrain.py \
    --dataset_path data/multimodal_dataset.pt \
    --vocab_path_raw models/vocab_raw.txt \
    --vocab_path_size models/vocab_size.txt \
    --pretrained_model_path models/multimodal_stage2-70000 \
    --output_model_path models/multimodal_stage2_phase2 \
    --config_path models/bert/base_config.json \
    --total_steps 30000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --warmup 0.1 \
    --target multimodal \
    --phase1_steps 0 \
    --balance_loss_alpha 0.1 \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --dist_train \
    --backend nccl
```

**Note**: In Phase 2:
- Remove `--freeze_encoders` flag
- Set `--phase1_steps 0` (already past phase 1)
- Load from Phase 1 checkpoint with `--pretrained_model_path`

### Automatic Two-Phase Training (Single Run)

The trainer will automatically transition from Phase 1 to Phase 2:

```bash
python pre-training/pretrain.py \
    --dataset_path data/multimodal_dataset.pt \
    --vocab_path_raw models/vocab_raw.txt \
    --vocab_path_size models/vocab_size.txt \
    --pretrained_raw_path models/raw_encoder_30000.bin \
    --pretrained_size_path models/size_encoder_30000.bin \
    --output_model_path models/multimodal_stage2 \
    --config_path models/bert/base_config.json \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --warmup 0.1 \
    --target multimodal \
    --freeze_encoders \
    --phase1_steps 70000 \
    --balance_loss_alpha 0.1 \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --dist_train \
    --backend nccl
```

**What happens:**
- Steps 1-70,000: Encoders frozen, fusion trains (Phase 1)
- At step 70,001: Automatically unfreezes encoders
- Steps 70,001-100,000: Joint training (Phase 2)

---

## Key Arguments

### Required Arguments

- `--dataset_path`: Path to multi-modal dataset
- `--vocab_path_raw`: Vocabulary for Raw Packet modality
- `--vocab_path_size`: Vocabulary for Packet Size modality
- `--pretrained_raw_path`: Pretrained Raw encoder checkpoint (Stage 1)
- `--pretrained_size_path`: Pretrained Size encoder checkpoint (Stage 1)
- `--output_model_path`: Where to save checkpoints
- `--target multimodal`: Must specify multimodal target
- `--total_steps`: Total training steps (100,000 recommended)

### Multi-Modal Specific Arguments

- `--freeze_encoders`: Freeze encoders in Phase 1 (use this flag for Phase 1)
- `--phase1_steps`: When to transition to Phase 2 (default: 70,000)
- `--balance_loss_alpha`: Balance loss weight (default: 0.1)

### Additional Arguments

- `--corpus_path_raw`: Raw corpus path (if building dataset on-the-fly)
- `--corpus_path_size`: Size corpus path (if building dataset on-the-fly)
- `--dynamic_masking`: Enable dynamic masking (recommended)
- `--clip_grad_norm`: Gradient clipping (default: 1.0)

---

## Expected Training Logs

### Phase 1 (Frozen Encoders)

```
|    10000/  100000 steps | Phase1 |   120.5 s |  2048.32 tokens/s | loss   3.45 | cmm: 0.693 | cmmp: 2.500 | bal: 0.250 | acc_cmm: 0.523 | acc_cmmp: 0.145 | g_raw: 0.520 | g_size: 0.480
|    20000/  100000 steps | Phase1 |   118.2 s |  2088.12 tokens/s | loss   2.85 | cmm: 0.512 | cmmp: 2.100 | bal: 0.230 | acc_cmm: 0.742 | acc_cmmp: 0.245 | g_raw: 0.510 | g_size: 0.490
...
```

### Phase Transition

```
================================================================================
TRANSITIONING TO PHASE 2: Unfreezing encoders
================================================================================
```

### Phase 2 (Joint Training)

```
|    70100/  100000 steps | Phase2 |   125.3 s |  1965.12 tokens/s | loss   2.12 | cmm: 0.312 | cmmp: 1.650 | bal: 0.150 | acc_cmm: 0.865 | acc_cmmp: 0.425 | g_raw: 0.505 | g_size: 0.495
|    80000/  100000 steps | Phase2 |   122.8 s |  2010.45 tokens/s | loss   1.85 | cmm: 0.265 | cmmp: 1.450 | bal: 0.120 | acc_cmm: 0.892 | acc_cmmp: 0.512 | g_raw: 0.502 | g_size: 0.498
...
```

### Metrics Explanation

- **loss**: Total loss (CMM + CMMP + 0.1*Balance)
- **cmm**: CMM task loss (binary cross-entropy)
- **cmmp**: CMMP task loss (cross-entropy)
- **bal**: Balance loss (modality usage imbalance)
- **acc_cmm**: CMM accuracy (% correctly classified matches)
- **acc_cmmp**: CMMP accuracy (% correctly predicted masked tokens)
- **g_raw**: Average gate weight for Raw modality
- **g_size**: Average gate weight for Size modality

**Healthy training signs:**
- `g_raw` ≈ `g_size` ≈ 0.5 (balanced modality usage)
- `acc_cmm` increases (better cross-modal matching)
- `acc_cmmp` increases (better cross-modal prediction)
- `bal` decreases (less imbalance)

---

## Output Checkpoints

Checkpoints are saved every `--save_checkpoint_steps` (default: 10,000):

```
models/
  multimodal_stage2-10000
  multimodal_stage2-20000
  ...
  multimodal_stage2-70000  # End of Phase 1
  multimodal_stage2-80000
  ...
  multimodal_stage2-100000  # Final checkpoint
```

Each checkpoint contains:
- `embedding_raw`, `encoder_raw` (Raw Packet encoder)
- `embedding_size`, `encoder_size` (Packet Size encoder)
- `fusion` (Gated Multi-Modal Fusion)
- `target` (Multi-Modal Target)

---

## Troubleshooting

### Issue: "RuntimeError: vocab_raw not found"

**Solution**: Make sure you're passing both vocabularies:
```bash
--vocab_path_raw models/vocab_raw.txt \
--vocab_path_size models/vocab_size.txt
```

The script needs to be updated to handle two vocabularies. Check `pre-training/pretrain.py` and ensure it loads both:

```python
# Load Raw vocab
vocab_raw = Vocab()
vocab_raw.load(args.vocab_path_raw)
args.vocab_raw = vocab_raw.w2i

# Load Size vocab
vocab_size = Vocab()
vocab_size.load(args.vocab_path_size)
args.vocab_size = vocab_size.w2i

# Create tokenizers
args.tokenizer_raw = BertTokenizer(args, vocab_raw.w2i)
args.tokenizer_size = BertTokenizer(args, vocab_size.w2i)
```

### Issue: "Encoders are not frozen in Phase 1"

**Solution**: Ensure you're using the `--freeze_encoders` flag for Phase 1.

### Issue: "Gate weights heavily imbalanced (g_raw=0.9, g_size=0.1)"

**Possible causes:**
1. Balance loss weight too low → Increase `--balance_loss_alpha` (try 0.5)
2. One modality much easier to learn → Check data quality
3. Need more Phase 1 steps → Extend `--phase1_steps` to 80K

### Issue: "CMM accuracy stuck at 0.5 (random guessing)"

**Possible causes:**
1. Negative sampling too easy → Implement hard negative sampling
2. Encoders not learning cross-modal features → Increase Phase 2 steps
3. Fusion module too weak → Check architecture

### Issue: "CMMP accuracy very low (<0.1)"

**Possible causes:**
1. Raw→Size prediction too difficult → Check if there's correlation in your data
2. Size vocab too large → Verify vocabulary size is reasonable
3. Masking ratio too high → Reduce from 15% to 10%

---

## Next Steps After Stage 2

After completing Stage 2 pretraining:

1. **Fine-tuning** on downstream tasks (e.g., traffic classification)
2. **Evaluation** on test sets
3. **Analysis** of learned representations (t-SNE, attention visualization)
4. **Ablation studies** to understand component contributions

---

## Additional Notes

### Differential Learning Rates (Phase 2)

Currently, the code uses a single learning rate. To implement differential LR:

1. Modify optimizer initialization in `uer/trainer.py`:
```python
# In worker() function
if args.target == "multimodal" and current_step > phase1_steps:
    # Differential LR for Phase 2
    param_optimizer = list(model.named_parameters())
    fusion_params = ['fusion', 'target']
    encoder_params = ['embedding_raw', 'encoder_raw', 'embedding_size', 'encoder_size']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(fp in n for fp in fusion_params)],
         'lr': 2e-5},
        {'params': [p for n, p in param_optimizer if any(ep in n for ep in encoder_params)],
         'lr': 5e-6}
    ]
```

### Hard Negative Sampling

The current implementation uses random negative sampling. To enable hard negative sampling:

1. In `MultiModalDataLoader`, compute encoder features first
2. Use `hard_negative_sampling()` from `multimodal_target.py`
3. This requires an extra forward pass and may slow down training

**Trade-off**: Better CMM performance vs. training speed

### Memory Optimization

If you encounter OOM errors:

1. Reduce `--batch_size` (try 32 per GPU)
2. Enable gradient checkpointing
3. Use mixed precision training (`--fp16`)
4. Reduce sequence lengths (Raw: 512→384, Size: 256→192)

---

## References

- **MM4flow**: Cross-attention fusion design
- **BLIP**: Image-text multi-modal pretraining (CMM + ITM tasks)
- **Mixture of Experts (MOE)**: Gating mechanism
- **BERT**: Masked language modeling strategy
