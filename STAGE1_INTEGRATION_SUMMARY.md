# Stage 1 Multi-Modal Pre-training Integration Summary

## Overview

Successfully integrated Stage 1 single-modal pre-training for traffic analysis into the uer framework. This implementation supports two modalities:
1. **Raw Packet** - Bigram tokenization of packet payloads
2. **Packet Size** - Packet size sequences with direction encoding

## What Was Done

### 1. Vocabulary Files Created ✅

Generated vocabulary files for both modalities:
- `vocab_raw.txt`: 65,541 tokens (65,536 bigrams + 5 special tokens)
- `vocab_size.txt`: 3,006 tokens (3,001 size tokens + 5 special tokens)

**Special tokens (IDs 0-4):**
- [PAD] = 0
- [UNK] = 1
- [CLS] = 2
- [SEP] = 3
- [MASK] = 4

**Location:** Project root directory

### 2. Embedding Classes Added ✅

Added two new embedding classes to `uer/layers/embeddings.py`:

#### RawPacketEmbedding (lines 114-188)
```python
Formula: emb = token_emb + direction_emb + position_emb + protocol_emb

Components:
- Token embedding: Bigram tokens from vocab_raw.txt
- Direction embedding: ±1 → indices 0,2 (3 values total)
- Position embedding: Sequence position
- Protocol embedding: TCP=0, UDP=1 (broadcast across sequence)
```

#### PacketSizeEmbedding (lines 191-256)
```python
Formula: emb = size_emb + position_emb + protocol_emb

Components:
- Size embedding: Direction already encoded in token (size * direction + 1500)
- Position embedding: Sequence position
- Protocol embedding: TCP=0, UDP=1 (broadcast across sequence)
```

**Registered in:** `uer/layers/__init__.py`

### 3. Dataset & DataLoader Classes Added ✅

Added to `uer/utils/data.py` (lines 810-1183):

#### RawPacketDataset
- Reads `corpus_raw.txt` format
- Extracts protocol from first packet
- Handles direction information (currently default to +1)
- Supports static and dynamic masking

#### RawPacketDataLoader
- Batch format: `(src, tgt_mlm, directions, protocols)`
- Applies MLM masking (15% mask rate)

#### PacketSizeDataset
- Reads `corpus_size.txt` format
- Direction encoded in size tokens
- Protocol detection (default TCP)

#### PacketSizeDataLoader
- Batch format: `(src, tgt_mlm, protocols)`
- Applies MLM masking (15% mask rate)

**Registered in:** `uer/utils/__init__.py`

### 4. Trainer Classes Added ✅

Added to `uer/trainer.py` (lines 207-311):

#### RawPacketMlmTrainer
- Handles batch format from RawPacketDataLoader
- MLM-only training (no sentence prediction)
- Reports: loss_mlm, acc_mlm

#### PacketSizeMlmTrainer
- Handles batch format from PacketSizeDataLoader
- MLM-only training (no sentence prediction)
- Reports: loss_mlm, acc_mlm

**Registered in:** `str2trainer` dictionary

### 5. Model Classes Added ✅

Created `uer/models/stage1_models.py`:

#### RawPacketModel
- Combines: RawPacketEmbedding + Encoder + MlmTarget
- Weight tying: token_embedding ↔ mlm_linear_2
- Handles directions and protocol parameters
- Creates seg tensor (all 1s for Stage 1)

#### PacketSizeModel
- Combines: PacketSizeEmbedding + Encoder + MlmTarget
- Weight tying: size_embedding ↔ mlm_linear_2
- Handles protocol parameter
- Creates seg tensor (all 1s for Stage 1)

**Integrated in:** `uer/model_builder.py`

### 6. Target Registration ✅

Updated `uer/targets/__init__.py`:
- Added `"raw_packet": MlmTarget`
- Added `"packet_size": MlmTarget`

Updated `pre-training/pretrain.py`:
- Added `raw_packet` and `packet_size` to target choices

## Complete Training Workflow

### Step 1: Generate Text Corpus

First, extract features from PCAP files:

```bash
python data_generation/multimodal_data_gen_v3.py \
    --pcap_dir ./pcaps \
    --output_raw corpus_raw.txt \
    --output_size corpus_size.txt \
    --output_meta metadata.json \
    --bytes_per_packet 64 \
    --max_raw_packets 8
```

**Input:** PCAP files named as `{protocol}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap`

**Output:**
- `corpus_raw.txt`: Raw packet bigrams (hex, space-separated)
  ```
  ||4500 0006 0683 3c52 5297 ...

  0000 003c 3c52 ...

  ||...
  ```

- `corpus_size.txt`: Packet sizes (direction-encoded)
  ```
  ||1672 2185 953 ...

  1234 567 ...

  ||...
  ```

### Step 2: Preprocess Dataset (Raw Packet)

```bash
python pre-training/preprocess.py \
    --corpus_path corpus_raw.txt \
    --vocab_path vocab_raw.txt \
    --dataset_path dataset_raw.pt \
    --target raw_packet \
    --seq_length 512 \
    --processes_num 4 \
```

**Output:** `dataset_raw.pt` (pickled instances)

### Step 3: Preprocess Dataset (Packet Size)

```bash
python pre-training/preprocess.py \
    --corpus_path corpus_size.txt \
    --vocab_path vocab_size.txt \
    --dataset_path dataset_size.pt \
    --target packet_size \
    --seq_length 512 \
    --processes_num 4 \
```

**Output:** `dataset_size.pt` (pickled instances)

### Step 4: Train Raw Packet Model (Stage 1)

```bash
python pre-training/pretrain.py \
    --dataset_path dataset_raw.pt \
    --vocab_path vocab_raw.txt \
    --output_model_path models/stage1_raw.bin \
    --config_path models/bert/base_config.json \
    --embedding raw_packet \
    --encoder transformer \
    --target raw_packet \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --warmup 0.1 \
    --dropout 0.1 \
    --seed 42
```

**Key Arguments:**
- `--embedding raw_packet`: Use RawPacketEmbedding
- `--target raw_packet`: Use RawPacketMlmTrainer + RawPacketModel + MlmTarget
- `--encoder transformer`: Use standard transformer encoder

**Output:**
- `models/stage1_raw.bin`: Pre-trained raw packet encoder
- `models/stage1_raw.bin-{steps}`: Intermediate checkpoints

### Step 5: Train Packet Size Model (Stage 1)

```bash
python pre-training/pretrain.py \
    --dataset_path dataset_size.pt \
    --vocab_path vocab_size.txt \
    --output_model_path models/stage1_size.bin \
    --config_path models/bert/base_config.json \
    --embedding packet_size \
    --encoder transformer \
    --target packet_size \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 100000 \
    --save_checkpoint_steps 10000 \
    --report_steps 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --warmup 0.1 \
    --dropout 0.1 \
    --seed 42
```

**Key Arguments:**
- `--embedding packet_size`: Use PacketSizeEmbedding
- `--target packet_size`: Use PacketSizeMlmTrainer + PacketSizeModel + MlmTarget

**Output:**
- `models/stage1_size.bin`: Pre-trained packet size encoder
- `models/stage1_size.bin-{steps}`: Intermediate checkpoints

## Architecture Summary

### Raw Packet Modality
```
Input: Bigram tokens (hex) + Directions + Protocol
       ↓
RawPacketEmbedding:
  - Token embedding (vocab_size=65541, emb_size=768)
  - Direction embedding (3 values, emb_size=768)
  - Position embedding (max_seq_len=512, emb_size=768)
  - Protocol embedding (2 values, emb_size=768)
       ↓ (element-wise addition)
  Combined embedding [batch, seq_len, 768]
       ↓
TransformerEncoder:
  - 12 layers, 12 heads, hidden_size=768
  - Self-attention with fully_visible mask
       ↓
MlmTarget:
  - Linear projection: hidden_size → emb_size
  - LayerNorm
  - Linear projection: emb_size → vocab_size
  - Softmax + NLLLoss
       ↓
Output: loss_mlm, correct_mlm, denominator
```

### Packet Size Modality
```
Input: Size tokens (direction-encoded) + Protocol
       ↓
PacketSizeEmbedding:
  - Size embedding (vocab_size=3006, emb_size=768)
  - Position embedding (max_seq_len=512, emb_size=768)
  - Protocol embedding (2 values, emb_size=768)
       ↓ (element-wise addition)
  Combined embedding [batch, seq_len, 768]
       ↓
TransformerEncoder:
  - Same as raw packet
       ↓
MlmTarget:
  - Same as raw packet
       ↓
Output: loss_mlm, correct_mlm, denominator
```

## Corpus Format Details

### corpus_raw.txt
```
||{first_packet_bigrams}

{packet_2_bigrams}

{packet_3_bigrams}

||{next_flow_first_packet}
...
```

- Flow separator: `||` at beginning of line
- Packet separator: blank line
- Bigrams: space-separated hex strings (e.g., "4500 0006 0683")
- First packet used for protocol detection (byte offset 9 in IP header)

### corpus_size.txt
```
||{size_tokens_space_separated}

||{next_flow}
...
```

- Flow separator: `||` at beginning of line
- Flow terminator: blank line
- Size tokens: `size * direction + 1500`
  - Example: 100-byte uplink → 100 * 1 + 1500 = 1600
  - Example: 200-byte downlink → 200 * (-1) + 1500 = 1300

## Data Format Compatibility

The implementation is **fully compatible** with uer's existing preprocess.py:

1. **Text corpus format**: Space-separated tokens with flow separators
2. **Vocab format**: One token per line with special tokens at top
3. **Dataset format**: Pickled instances in `.pt` files
4. **Masking**: Standard 15% mask rate with 80/10/10 split
5. **Special tokens**: [PAD], [UNK], [CLS], [SEP], [MASK]

## Validation Checklist

✅ Vocab files created successfully
✅ Embedding classes registered in uer framework
✅ Dataset/DataLoader classes integrated
✅ Trainer classes added for both modalities
✅ Model classes created with proper interfaces
✅ Target registration completed
✅ Model builder updated to use Stage1 models
✅ Pretrain.py updated with new targets
✅ Encoder compatibility ensured (seg tensor generation)
✅ Weight tying configured correctly

## Next Steps for Stage 2

Stage 2 will involve:
1. **Cross-attention fusion** between raw and size modalities
2. **SimSiam contrastive learning** for representation alignment
3. **Cross-modal MLM** for learning joint representations

Components needed:
- CrossAttentionFusion module (MM4flow-inspired)
- SimSiam projection heads
- Joint model combining both encoders
- Multi-modal dataset/dataloader for paired inputs

## File Changes Summary

**New Files:**
- `vocab_raw.txt` (65,541 tokens)
- `vocab_size.txt` (3,006 tokens)
- `uer/models/stage1_models.py` (RawPacketModel, PacketSizeModel)
- `data_generation/create_vocab_multimodal.py`
- `data_generation/multimodal_data_gen_v3.py`

**Modified Files:**
- `uer/layers/embeddings.py` (+143 lines: RawPacketEmbedding, PacketSizeEmbedding)
- `uer/layers/__init__.py` (registered new embeddings)
- `uer/utils/data.py` (+374 lines: 4 new Dataset/DataLoader classes)
- `uer/utils/__init__.py` (registered new datasets/dataloaders)
- `uer/trainer.py` (+105 lines: 2 new trainer classes)
- `uer/targets/__init__.py` (added raw_packet, packet_size targets)
- `uer/model_builder.py` (integrated Stage1 models)
- `pre-training/pretrain.py` (added target choices)

## Usage Examples

### Quick Test (CPU mode)

```bash
# Generate small test corpus (assuming you have test PCAPs)
python data_generation/multimodal_data_gen_v3.py \
    --pcap_dir ./test_pcaps \
    --output_raw test_corpus_raw.txt \
    --output_size test_corpus_size.txt

# Preprocess
python pre-training/preprocess.py \
    --corpus_path test_corpus_raw.txt \
    --vocab_path vocab_raw.txt \
    --dataset_path test_dataset_raw.pt \
    --target raw_packet \
    --seq_length 128 \
    --processes_num 1

# Train (CPU, small steps for testing)
python pre-training/pretrain.py \
    --dataset_path test_dataset_raw.pt \
    --vocab_path vocab_raw.txt \
    --output_model_path test_model.bin \
    --config_path models/bert/base_config.json \
    --embedding raw_packet \
    --target raw_packet \
    --total_steps 100 \
    --batch_size 8 \
    --report_steps 10
```

### Multi-GPU Training

```bash
python pre-training/pretrain.py \
    --dataset_path dataset_raw.pt \
    --vocab_path vocab_raw.txt \
    --output_model_path models/stage1_raw_distributed.bin \
    --config_path models/bert/base_config.json \
    --embedding raw_packet \
    --target raw_packet \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --master_ip tcp://localhost:12345 \
    --backend nccl \
    --batch_size 32 \
    --total_steps 100000
```

## Expected Training Metrics

During training, you should see output like:

```
|     100/  100000 steps|  12.345 s| 41234.56 tokens/s| loss_mlm:   8.234| acc_mlm: 0.125
|     200/  100000 steps|  11.234 s| 43567.89 tokens/s| loss_mlm:   7.891| acc_mlm: 0.156
|     300/  100000 steps|  10.987 s| 44123.45 tokens/s| loss_mlm:   7.456| acc_mlm: 0.189
...
```

**Initial metrics (random initialization):**
- loss_mlm: 8-10 (log(vocab_size) ≈ 11 for raw, 8 for size)
- acc_mlm: 0.10-0.15 (slightly better than random)

**Converged metrics (after full training):**
- loss_mlm: 2-4 (depends on task difficulty)
- acc_mlm: 0.30-0.50 (30-50% of masked tokens predicted correctly)

## Troubleshooting

### Issue: "No such file or directory: /tmp/dataset-raw-tmp-0.pt"

**Cause:** Dataset worker writes to /tmp but directory doesn't exist or has permissions issues.

**Solution:** Modify `uer/utils/data.py` lines 845 and 1054 to use a writable path:
```python
dataset_writer = open("./temp/dataset-raw-tmp-" + str(proc_id) + ".pt", "wb")
```

### Issue: "IndexError: list index out of range" in protocol detection

**Cause:** First packet too short or unexpected format.

**Solution:** Check PCAP format and adjust protocol detection logic in Dataset worker() method.

### Issue: "RuntimeError: CUDA out of memory"

**Solutions:**
- Reduce `--batch_size`
- Reduce `--seq_length`
- Reduce `--instances_buffer_size`
- Use gradient accumulation: `--accumulation_steps 2`

### Issue: Low MLM accuracy (<10%)

**Possible causes:**
- Vocab file doesn't match corpus tokens
- Tokenizer not parsing correctly
- Check vocab IDs: token "4500" should map to ID 17669 (17664 + 5)

## References

- **TrafficFormer**: Original bigram tokenization approach
- **MM4flow**: Multi-modal fusion and SimSiam inspiration
- **UER-py**: Framework for BERT-style pre-training
- **BERT**: MLM pre-training task
