# MM-TrafficBERT

**MM-TrafficBERT: Multimodal Pre-training with Deep Reliability-Aware Fusion for Network Traffic Analysis**

This repository hosts the open-science release of MM-TrafficBERT: source code, vocabularies, and configs for reproducing the two-stage pre-training and downstream fine-tuning reported in the paper.

---

## 1. Repository layout

```
.
├── data_generation/
│   ├── pcap_process.cpp          C++ flow splitter (raw pcap → per-flow pcap)
│   ├── Makefile                  build pcap_process
│   └── multimodal_data_gen.py    flow pcap → (corpus_raw.txt, corpus_size.txt)
├── pre-training/
│   ├── preprocess.py             text corpus → .pt dataset (Stage 1 / Stage 2)
│   └── pretrain.py               Stage 1 unimodal MLM / Stage 2 multimodal
├── fine-tuning/
│   ├── multimodal_data_utils.py  per-flow pcap → train/val/test pickles
│   ├── run_classifier_stage2.py  Stage 2 fine-tune (full multimodal + ITGCA)
│   ├── run_classifier_stage1.py  Stage 1 fine-tune (single-modality baseline)
│   └── run_inference.py          held-out test inference
├── uer/                       trimmed UER-py / ET-BERT backbone
├── models/bert/
│   ├── base_config.json           12-layer config (content encoder)
│   ├── behavior_6_config.json     6-layer  config (behavior encoder)
│   ├── vocab_raw.txt              byte vocab  (256 + 5 special)
│   ├── vocab_size.txt             size vocab  (3001 + 5 special)
│   └── vocab_temporal.txt         IAT  vocab  (1000 + 5 special)
└── requirements.txt
```

The three vocabularies under `models/bert/` are deterministic and shipped with the release; you do **not** need to regenerate them.

---

## 2. Released artifacts (Zenodo)

The Stage 2 multimodal pretrained checkpoint and the per-dataset fine-tuning splits used in the paper are hosted on Zenodo, because anonymous.4open.science enforces a per-file size limit too small for these artifacts. Anonymous deposit:

    https://doi.org/10.5281/zenodo.19992866

Layout of the deposit:

```
zenodo_release/
├── pretrain.bin                       Stage 2 multimodal pretrained checkpoint
└── datasets/
    ├── AnonProxy/                     train.pkl, val.pkl, test.pkl, label2id.pkl
    ├── Appsniffer/                      (same four files in each subdirectory)
    ├── cstnet-tls/
    ├── datacon2021-p2/
    ├── browser/
    └── ITC-Net-Blend/
```

To follow the commands in §5 / §6 verbatim, unpack the deposit so that:

```
project/models/mm_trafficbert.bin                                    ← from pretrain.bin
project/data/finetune/<task>/processed/{train,val,test,label2id}.pkl ← from datasets/<dataset_name>/
```

Equivalently, leave the layout flat and pass absolute paths to the scripts.

The pickles are pre-tokenized with the byte / size / temporal vocabularies under `models/bert/`. Reviewers who want to verify downstream numbers without redoing pre-training (§4) can start directly from §5 (fine-tuning) using these artifacts.

---

## 3. Setup

Hardware used in the paper: 4 × NVIDIA A100 80 GB. Software: PyTorch 2.9.1 + CUDA 13.0.

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
# Build the C++ flow splitter (requires libpcap-dev)
cd data_generation && make && cd ..
```

`pcap_process` outputs one `.pcap` per (TCP|UDP) 5-tuple flow under the file-name format
`{TCP|UDP}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap`,
and discards flows with fewer than 5 payload-bearing packets (consistent with the paper's filter).

---

## 4. Pre-training

End-to-end pipeline:

```
raw pcap dir  ──► [pcap_process]  ──►  per-flow pcap dir
              ──► [multimodal_data_gen.py]  ──►  corpus_raw.txt + corpus_size.txt
              ──► [preprocess.py]  ──►  Stage 1 dataset.pt   ──► [pretrain.py --target raw_packet ]  ──► raw_encoder.bin
                                        Stage 1 dataset.pt   ──► [pretrain.py --target packet_size]  ──► size_encoder.bin
                                        Stage 2 dataset.pt   ──► [pretrain.py --target multimodal ]  ──► mm_trafficbert.bin
```

### 4.1 Split pcaps into per-flow pcaps

Put **all** raw pcaps anywhere under one directory tree (subdirectories OK — `pcap_process` recursively scans). For pre-training use the **default mode** (no `-l`), which writes one output sub-directory per input pcap:

```bash
./data_generation/pcap_process /path/to/pretrain_pcaps/ /path/to/pretrain_flows/
```

### 4.2 Build the text corpora

```bash
python3 data_generation/multimodal_data_gen.py \
    --pcap_dir         /path/to/pretrain_flows/ \
    --output_raw       data/corpus_raw.txt \
    --output_size      data/corpus_size.txt \
    --num_workers      32
```

### 4.3 Preprocess into `.pt`

```bash
# Stage 1 — content (raw bytes, 12-layer encoder)
python3 pre-training/preprocess.py \
    --target raw_packet \
    --corpus_path  data/corpus_raw.txt \
    --vocab_path   models/bert/vocab_raw.txt \
    --dataset_path data/raw_dataset.pt \
    --seq_length   512 \
    --processes_num 32 \
    --dynamic_masking

# Stage 1 — behavior (size + IAT, 6-layer encoder)
python3 pre-training/preprocess.py \
    --target packet_size \
    --corpus_path         data/corpus_size.txt \
    --vocab_path          models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --dataset_path        data/size_dataset.pt \
    --seq_length          256 \
    --processes_num       32 \
    --dynamic_masking

# Stage 2 — paired multimodal
python3 pre-training/preprocess.py \
    --target multimodal \
    --corpus_path_raw     data/corpus_raw.txt \
    --corpus_path_size    data/corpus_size.txt \
    --vocab_path_raw      models/bert/vocab_raw.txt \
    --vocab_path_size     models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --dataset_path        data/mm_dataset.pt \
    --seq_length_raw      512 \
    --seq_length_size     256 \
    --processes_num       32 \
    --dynamic_masking
```

### 4.4 Stage 1 — unimodal pre-training

```bash
# Content encoder (12 layers)
python3 pre-training/pretrain.py \
    --target raw_packet \
    --dataset_path       data/raw_dataset.pt \
    --vocab_path         models/bert/vocab_raw.txt \
    --output_model_path  models/raw_encoder.bin \
    --config_path        models/bert/base_config.json \
    --total_steps        100000 \
    --batch_size         64 \
    --learning_rate      5e-5 \
    --world_size 4 --gpu_ranks 0 1 2 3 \
    --master_ip tcp://localhost:12345

# Behavior encoder (6 layers; uses DualMlm with size CE + IAT soft-label KL, σ=10)
python3 pre-training/pretrain.py \
    --target packet_size \
    --dataset_path        data/size_dataset.pt \
    --vocab_path          models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --output_model_path   models/size_encoder.bin \
    --config_path         models/bert/behavior_6_config.json \
    --total_steps         100000 \
    --batch_size          64 \
    --learning_rate       5e-5 \
    --world_size 4 --gpu_ranks 0 1 2 3 \
    --master_ip tcp://localhost:12345
```

### 4.5 Stage 2 — multimodal pre-training

```bash
python3 pre-training/pretrain.py \
    --target multimodal \
    --dataset_path        data/mm_dataset.pt \
    --vocab_path_raw      models/bert/vocab_raw.txt \
    --vocab_path_size     models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --pretrained_raw_path  models/raw_encoder.bin \
    --pretrained_size_path models/size_encoder.bin \
    --output_model_path    models/mm_trafficbert.bin \
    --config_path          models/bert/base_config.json \
    --config_path_size     models/bert/behavior_6_config.json \
    --use_itgca  \
    --num_fusion_layers 6  \
    --total_steps        100000 \
    --batch_size         64 \
    --learning_rate      5e-5 \
    --encoder_lr_ratio   0.3 \
    --seq_length_raw 512 --seq_length_size 256 \
    --world_size 4 --gpu_ranks 0 1 2 3 \
    --master_ip tcp://localhost:12345
```

### 4.6 OOM mitigation: gradient accumulation

Stage 2 is the most memory-hungry step (two encoders + 6-layer fusion + momentum encoders + 4096-entry queue + ITM hard negatives that triple the fused-CLS forward pass). On smaller GPUs, reduce `--batch_size` and compensate with `--accumulation_steps` to keep the optimizer batch fixed:

| Global optimizer batch | Per-GPU batch | `--accumulation_steps` (4 GPUs) |
|------------------------|---------------|---------------------------------|
| 256 (paper)     | 64            | 1                               |
| 256             | 32            | 2                               |
| 256             | 16            | 4                               |
| 256             |  8            | 8                               |

For Stage 2, gradient accumulation does not enlarge the per-forward ITC/ITM candidate set: `--batch_size` remains the local contrastive/matching batch, while the table above preserves only the optimizer batch. The momentum teacher and feature queue remain fixed throughout each accumulation group and are updated once after its optimizer step. Stage 1 MLM has a separable per-example objective, so accumulation is equivalent to a larger optimizer batch up to normal stochastic differences.

### Component-level ablation flags

Disable individual ITGCA components to reproduce the per-component decomposition (Table 5):

| Flag                  | Disables                                                  |
|-----------------------|-----------------------------------------------------------|
| `--ablate_r_stat`     | Flow-level Shannon-entropy prior (Eq. 5)                  |
| `--ablate_g_token`    | Per-position learned token gate (Eq. 8)                   |
| `--ablate_source_bias`| Source-side attention reweighting (Eq. 10)                |

These flags must be passed identically at fine-tune time so that loaded checkpoint geometry matches.

---

## 5. Fine-tuning

End-to-end pipeline:

```
labelled pcap dir  ──► [pcap_process -l]  ──►  per-label flow pcap dir
                  ──► [multimodal_data_utils.py]  ──►  train.pkl, val.pkl, test.pkl, label2id.pkl
                  ──► [run_classifier_stage2.py]   ──►  finetuned classifier
```

### 5.1 Split pcaps, preserving label directories

The fine-tuning input is a directory whose immediate sub-directories are class names:

```
finetune_input/
├── label_a/  *.pcap
├── label_b/  *.pcap
└── label_c/  *.pcap
```

Use **label mode** so that the output preserves the same class layout:

```bash
./data_generation/pcap_process -l /path/to/finetune_input/ /path/to/finetune_flows/
```

### 5.2 Build train/val/test pickles

```bash
python3 fine-tuning/multimodal_data_utils.py \
    --pcap_dir            /path/to/finetune_flows/ \
    --vocab_path_raw      models/bert/vocab_raw.txt \
    --vocab_path_size     models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --output_dir          data/finetune/<task>/processed/ \
    --max_samples_per_class 500 \
    --seed 42 \
    --num_workers 32 
```

Outputs four files in `--output_dir`: `train.pkl`, `val.pkl`, `test.pkl`, `label2id.pkl`. The split is per-class stratified at the flow level so that all packets of the same flow stay in one split.

### 5.3 Stage 2 multimodal fine-tuning

The paper uses two-phase training: **Phase 1** (5 epochs) freezes the backbone and warms up the classifier head; **Phase 2** (10 epochs) unfreezes everything with layer-wise LR decay (encoder 0.3 ×, fusion 0.7 ×, classifier 1 ×). Base LR 5e-5, AdamW.

```bash
python3 fine-tuning/run_classifier_stage2.py \
    --train_path      data/finetune/<task>/processed/train.pkl \
    --dev_path        data/finetune/<task>/processed/val.pkl \
    --test_path       data/finetune/<task>/processed/test.pkl \
    --label2id_path   data/finetune/<task>/processed/label2id.pkl \
    --vocab_path_raw      models/bert/vocab_raw.txt \
    --vocab_path_size     models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --pretrained_model_path models/mm_trafficbert.bin \
    --output_model_path     models/<task>_classifier.bin \
    --config_path           models/bert/base_config.json \
    --config_path_size      models/bert/behavior_6_config.json \
    --use_itgca  \
    --batch_size 32 --learning_rate 5e-5 \
    --phase1_epochs 5 --phase1_lr 1e-3 \
    --epochs_num 10 \
    --llrd_encoder_ratio 0.3 --llrd_fusion_ratio 0.7 \
    --label_smoothing 0.1 --max_grad_norm 1.0 \
    --seed 42 --gpu_ranks 0
```

Important: every `--use_itgca / --ablate_*` / `--num_fusion_layers` / `--itgca_window_size` flag here **must match the value used at Stage 2 pre-training** — otherwise the gate parameters in the checkpoint cannot be loaded. The classifier prints a warning if it detects a mismatch.

### 5.4 Few-shot fine-tuning

Pass `--few_shot 0.1` (or 0.4 / 0.7) to use a stratified sub-sample of the training set, matching Section 5.3 of the paper. Validation and test sets are not affected.

---

## 6. Inference and analysis

```bash
# Standard inference on a held-out test pickle
python3 fine-tuning/run_inference.py \
    --load_model_path  models/<task>_classifier.bin \
    --test_path        data/finetune/<task>/processed/test.pkl \
    --label2id_path    data/finetune/<task>/processed/label2id.pkl \
    --vocab_path_raw      models/bert/vocab_raw.txt \
    --vocab_path_size     models/bert/vocab_size.txt \
    --vocab_path_temporal models/bert/vocab_temporal.txt \
    --config_path         models/bert/base_config.json \
    --config_path_size    models/bert/behavior_6_config.json \
    --use_itgca --num_fusion_layers 6
```

---


## 7. Acknowledgement

This codebase is built on top of [UER-py](https://github.com/dbiir/UER-py) . We thank the authors for releasing high-quality reference implementations.
