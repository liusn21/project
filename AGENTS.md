# MM-TrafficBERT Project Notes

These notes are durable context for future Codex sessions in this repository.
They should describe stable project structure, research intent, code flow, and
collaboration constraints. Do not use this file as a per-chat log.

## Collaboration Constraints

- The project is not run on this local machine. Do not start training, testing,
  inference, preprocessing, data generation, or LaTeX builds locally unless the
  user explicitly asks for it.
- Static inspection commands such as listing files and reading source are fine.
- Do not modify any project file unless the user explicitly authorizes an edit.
  Requests to read, inspect, discuss, review, propose, or plan changes do not
  constitute edit authorization; present proposed changes in chat and wait for
  approval before applying them.
- The user mainly wants help with two kinds of work:
  1. Modify and improve the project code and paper.
  2. Develop follow-up research ideas extending this project.
- Maintain critical thinking during collaboration. Treat the user's prompts as
  hypotheses, preferences, or proposed directions rather than facts to accept
  automatically. Check them against the code, manuscript, experiments, and
  research logic; point out mistakes, weak assumptions, risks, and better
  alternatives directly when they matter.
- Prefer editing source files and manuscript sources. Avoid modifying generated
  LaTeX/build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.bbl`,
  or generated PDFs unless explicitly requested.
- Be careful with existing user changes in the working tree. Never revert files
  the user changed unless explicitly asked.

## Project Summary

MM-TrafficBERT is a multimodal pre-training framework for network traffic
analysis. The core paper title is:

`MM-TrafficBERT: Multimodal Pre-training with Deep Reliability-Aware Fusion for
Network Traffic Analysis`

The main idea is to combine two complementary traffic modalities:

- Content modality: raw transport payload bytes from early packets.
- Behavior modality: packet sizes, directions, and inter-arrival times (IATs).

The method uses a two-stage pre-training design:

- Stage 1 pre-trains each modality independently with MLM-style objectives.
- Stage 2 aligns the two modalities, then performs deep cross-modal fusion with
  Information-Theoretic Gated Cross-Attention (ITGCA).

The paper's main claim is that deep reliability-aware fusion during pre-training
outperforms content-only pre-training, behavior-only models, shallow input
concatenation, and late fine-tuning fusion. The reliability mechanism is designed
for encrypted traffic, where payload bytes may become high-entropy and
uninformative while behavior signals remain useful.

## Repository Map

- `README.md`: public reproduction guide and high-level pipeline.
- `data_generation/`: raw pcap processing and corpus generation.
  - `pcap_process.cpp`: C++ flow splitter, raw pcaps to per-flow pcaps.
  - `multimodal_data_gen.py`: per-flow pcaps to paired raw/size text corpora.
  - `Makefile`: builds `pcap_process` with libpcap.
- `pre-training/`: dataset preprocessing and pre-training entry points.
  - `preprocess.py`: text corpora to `.pt` datasets for Stage 1 or Stage 2.
  - `pretrain.py`: trains raw, size/IAT, or multimodal targets.
  - `count_params.py`: parameter-count helper.
- `fine-tuning/`: downstream data processing, fine-tuning, and inference.
  - `multimodal_data_utils.py`: labeled flow pcaps to train/val/test pickles.
  - `run_classifier_stage1.py`: single-modality or concat baseline classifier.
  - `run_classifier_stage2.py`: full multimodal classifier with fusion/ITGCA.
  - `run_inference.py`: held-out test inference and detailed metrics.
  - `measure_perflow.py`: per-flow measurement/analysis utility.
- `uer/`: trimmed UER/ET-BERT-style framework and MM-TrafficBERT internals.
  - `layers/embeddings.py`: raw and packet-size/IAT embeddings.
  - `layers/multimodal_fusion.py`: bidirectional fusion and ITGCA.
  - `models/multimodal_model.py`: Stage 2 ALBEF-style model.
  - `targets/multimodal_target.py`: ITC, ITM/CMM, masked reconstruction losses.
  - `utils/data.py`: Stage 1 and Stage 2 `.pt` dataset builders/loaders.
  - `trainer.py`: Stage 1 and Stage 2 training loops.
  - `model_builder.py`: constructs model variants from args/configs.
- `models/bert/`: shipped BERT configs and deterministic vocabularies.
  - `base_config.json`: 12-layer, hidden 768, 12-head config.
  - `behavior_6_config.json`: 6-layer behavior encoder config.
  - `vocab_raw.txt`: byte vocabulary.
  - `vocab_size.txt`: packet-size vocabulary.
  - `vocab_temporal.txt`: IAT vocabulary.
- `latex/ACM_conference/`: current paper source.
  - `main.tex`: ACM/CCS-style root file.
  - `sections/*.tex`: main manuscript sections.
  - `references.bib`: bibliography.
  - `figures/`: manuscript figures and plotting helpers.
- `followup_final.md`: merged design document for a follow-up robustness paper.
- `papers/`: related PDFs, research notes, and extracted presentation materials.
- `test/`: experiment notes, figures, result summaries, and older planning docs.
- `revision/`: revision/diagnostic scripts, including gate and entropy analyses.

## Data Flow

The intended end-to-end pipeline is:

1. Raw pcap directory.
2. `data_generation/pcap_process.cpp` splits traffic into per-flow pcaps.
3. `data_generation/multimodal_data_gen.py` converts per-flow pcaps into:
   - `corpus_raw.txt`
   - `corpus_size.txt`
4. `pre-training/preprocess.py` converts corpora into `.pt` datasets.
5. Stage 1 pre-training:
   - `--target raw_packet` trains the content encoder.
   - `--target packet_size` trains the behavior encoder.
6. Stage 2 pre-training:
   - `--target multimodal` loads Stage 1 encoders and trains alignment/fusion.
7. Fine-tuning:
   - labeled pcap folders are converted to pickles by
     `fine-tuning/multimodal_data_utils.py`.
   - `run_classifier_stage2.py` fine-tunes the full multimodal model.
8. Inference:
   - `run_inference.py` evaluates a fine-tuned classifier on a test pickle.

For pre-training, `pcap_process` default mode writes one output subdirectory per
input pcap. For fine-tuning, `pcap_process -l` preserves label directories.

`pcap_process.cpp` keeps TCP/UDP IPv4 flows, skips UDP DNS on port 53, preserves
the first observed direction as the flow filename direction, and filters out
flows with fewer than five payload-bearing packets. Output filenames have the
format:

`{TCP|UDP}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap`

## Tokenization And Inputs

Content/raw modality:

- Uses transport-layer payload only, not Ethernet/IP/TCP/UDP header fields.
- Reads up to the first 8 payload-bearing packets in a flow.
- Uses up to the first 64 payload bytes per selected packet.
- Byte-level tokens are hex strings, e.g. `45`, `00`, `ff`.
- Sequence format is `[CLS] + payload byte tokens + [SEP] + [PAD]`.
- Maximum raw sequence length is typically 512.
- Packet IDs are 0-7 for payload bytes and 8 for special/padding tokens.
- Direction IDs are 0 for downlink, 2 for uplink, and 1 for neutral
  special/padding tokens.
- The raw embedding is token + position + packet + direction.

Behavior modality:

- Uses up to 256 payload-bearing packets per flow.
- Packet size is clamped to `[0, 1500]`, multiplied by direction
  (`+1` upstream, `-1` downstream), then shifted by `+1500`.
- Size token range is therefore 0-3000.
- IAT tokenization uses:
  `int(sigmoid(log10(max(delta_t, 1e-6))) * 1000)`, clipped to 0-999.
- Size and IAT streams are aligned by packet position.
- Sequence format is `[CLS] + tokens + [SEP] + [PAD]`.
- Maximum behavior sequence length is typically 256.
- The behavior embedding is size token + temporal/IAT token + position.

The shipped vocabularies under `models/bert/` are deterministic and should not
be regenerated unless the user explicitly changes the tokenization design.

## Stage 1 Pre-Training

Stage 1 trains unimodal encoders independently:

- Raw/content encoder:
  - 12-layer BERT-style Transformer by default.
  - Standard MLM with 15 percent masking and BERT 80/10/10 replacement.
  - Implemented by `RawPacketDataset`, `RawPacketDataLoader`,
    `RawPacketModel`, and `RawPacketMlmTrainer`.
- Behavior encoder:
  - 6-layer BERT-style Transformer by default.
  - Synchronized masking: size and IAT positions are masked together.
  - Size prediction uses cross-entropy.
  - IAT prediction uses soft-label KL with Gaussian targets, default
    `sigma=10`, because adjacent IAT bins represent similar time intervals.
  - Implemented by `PacketSizeDataset`, `PacketSizeDataLoader`,
    `PacketSizeModel`, `DualMlmTarget`, and `PacketSizeMlmTrainer`.

The paper justifies the 6-layer behavior encoder because behavior tokens encode
lower-order size/timing statistics; the appendix reports that 12 behavior layers
do not improve average F1 and cost many more parameters.

## Stage 2 Multimodal Pre-Training

Stage 2 is implemented by `MultiModalModel`, `MultiModalFusionEncoder`, and
`MultiModalTarget`.

Architecture:

- Online raw embedding + encoder.
- Online size/IAT embedding + encoder.
- Momentum raw embedding + encoder.
- Momentum size/IAT embedding + encoder.
- Momentum projection heads for contrastive learning.
- FIFO feature queues for both modalities, default size 4096.
- Stacked bidirectional fusion layers, default 6.
- Target heads for ITC, ITM/CMM, and masked reconstruction.

Stage 2 inputs keep raw content clean/unmasked. Behavior size/IAT has clean
copies for ITC/ITM and masked copies for reconstruction.

Losses:

- ITC/CMC: symmetric contrastive alignment between raw CLS and size CLS.
  Uses projection heads, L2 normalization, default temperature 0.07, momentum
  teacher features, and queues.
- ITM/CMM: cross-modal matching on fused CLS pairs. It samples two hard negative
  types from the in-batch ITC similarity matrix.
- Masked reconstruction: reconstructs masked behavior size and IAT tokens from
  fused behavior features while raw content stays intact.
- Total Stage 2 loss is a weighted sum of ITC, ITM, size reconstruction, and
  temporal reconstruction. Defaults are all 1.0.

Important training details:

- ITC uses hard-label symmetric InfoNCE; there is no momentum-distillation
  soft-target mixture.
- `total_steps`, checkpoint suffixes, and report steps count micro-batches.
  The optimizer and scheduler update every `accumulation_steps` micro-batches.
- Momentum encoders/projections and feature queues update after every Stage 2
  micro-batch, matching the April compatibility behavior.
- When Stage 2 is built from Stage 1 checkpoints, `model_builder.py` loads the
  online and momentum encoders. No additional post-initialization hard sync of
  the online and momentum projection heads is performed.

## ITGCA

ITGCA means Information-Theoretic Gated Cross-Attention. It lives in
`uer/layers/multimodal_fusion.py` and is called from `MultiModalModel` and
`Stage2Classifier`.

The fusion layer is bidirectional:

- Raw branch: self-attention, then cross-attention with size as key/value, then
  FFN.
- Size branch: self-attention, then cross-attention with raw as key/value, then
  FFN.

The gate is asymmetric:

- Size <- Raw uses a content reliability prior because raw payload can degrade
  under encryption.
- Raw <- Size has no statistical prior and uses only learned compatibility.

Current implementation details:

- Flow-level raw reliability `r_stat` is computed from byte-token Shannon
  entropy in `compute_flow_reliability_raw()`.
- Local source reliability is computed by `compute_local_entropy()` with a
  default sliding window of 16.
- Learned compatibility is a bilinear CLS score:
  `sigmoid(c_q^T W c_k + b)`.
- When the statistical prior is active, raw reliability is first calibrated by
  `sigmoid(stat_scale * r_stat + stat_shift)`.
- The modality gate is:
  `r_mod = r_calibrated + sigmoid(alpha) * (r_learned - r_calibrated)`.
- Under `--ablate_r_learned`, Size <- Raw uses `r_calibrated` directly and
  Raw <- Size fixes its modality gate to 1, preserving the token gate and the
  cross-attention pathway.
- `alpha_init` defaults to -2.0, so the initial beta is about 0.12 and the model
  starts close to the statistical prior.
- The token gate is a learned per-position gate based on the self-attention
  residual. `token_gate_bias_init` defaults to +2.0, so the token gate starts
  mostly open.
- The final gate is multiplicative: `g = r_mod * g_token`.
- Source-side attention bias is used only for Size <- Raw. It is added to
  attention scores before scaling compensation and softmax.
- The cross-attention gate is applied after the attention output's final linear
  projection. This is intentional: if the gate goes to zero, the cross-modal
  contribution becomes zero and the residual keeps only self-attention features,
  avoiding attention-bias leakage.

ITGCA flags that affect checkpoint structure and must match between Stage 2
pre-training and fine-tuning:

- `--use_itgca`
- `--num_fusion_layers`
- `--itgca_window_size`
- `--ablate_r_stat`
- `--ablate_r_learned`
- `--ablate_g_token`
- `--ablate_source_bias`

`test/overview.md` contains older design notes and may disagree with the current
implementation. For example, older notes mention a `g_default` term, but the
current code and manuscript use `g = r_mod * g_token`.

## Fine-Tuning

Fine-tuning data is produced by `fine-tuning/multimodal_data_utils.py`.

- Input format is `pcap_dir/label_name/*.pcap`.
- Each pcap should already represent a bidirectional flow.
- The script extracts and tokenizes raw, size, and IAT modalities.
- It writes `train.pkl`, `val.pkl`, `test.pkl`, and `label2id.pkl`.
- Labels with fewer than `min_samples_per_class` samples are skipped.
- Classes are capped at `max_samples_per_class` for balancing.
- Default single-directory split is 80/10/10 at the flow level.
- A separate `--test_dir` mode is supported; labels must also exist in training.

Stage 2 classifier:

- Implemented by `fine-tuning/run_classifier_stage2.py`.
- Loads Stage 2 pretrained encoder/fusion weights and drops pre-training-only
  pieces: momentum modules, ITC projections, target heads, queues.
- Builds the same raw encoder, behavior encoder, and fusion module geometry.
- Classifier input is four hidden vectors concatenated:
  `raw_fused_CLS`, `size_fused_CLS`, raw non-CLS mean pool, size non-CLS mean
  pool.
- Classifier head is a two-layer MLP with Tanh and dropout.
- Phase 1 freezes the backbone and trains only the classifier head.
- Phase 2 unfreezes all parameters and uses differential learning rates:
  encoder at base LR times `llrd_encoder_ratio`, fusion at base LR times
  `llrd_fusion_ratio`, classifier at base LR.
- EMA is on during Phase 2. Optional FGM, multi-sample dropout, and R-Drop are
  available. MSD and R-Drop are mutually exclusive.
- Few-shot uses stratified subsampling of the training set only.

Stage 1 classifier:

- Implemented by `fine-tuning/run_classifier_stage1.py`.
- Supports `--modality raw`, `--modality size`, or `--modality both`.
- Uses CLS representations from the selected encoders.
- The `both` mode concatenates raw and size CLS without deep fusion.
- This is a baseline, not the full MM-TrafficBERT model.

## Manuscript Context

The current main paper source is under `latex/ACM_conference/`.

Core manuscript structure:

- `sections/01_introduction.tex`: motivation, observations, challenges,
  contributions.
- `sections/02_background.tex`: background.
- `sections/03_design.tex`: tokenization, Stage 1, Stage 2, ITGCA.
- `sections/04_evaluation.tex`: setup, baselines, classification, few-shot,
  ablation, ITGCA deep dive.
- `sections/05_discussion.tex`: limitations and future directions.
- `sections/06_related_work.tex`: related work.
- `sections/07_conclusion.tex`: conclusion.
- `sections/08_appendix.tex`: ethics, open science, datasets, baselines,
  leakage audit, sensitivity, shaping robustness, seed variability.

Current paper claims:

- MM-TrafficBERT is positioned as the first traffic-analysis framework to do
  deep cross-modal fusion during pre-training.
- It uses align-before-fuse: Stage 1 independent encoders, Stage 2 contrastive
  alignment before deep fusion.
- It introduces ITGCA, an asymmetric reliability-aware cross-attention mechanism
  using byte entropy as a training-free content reliability prior.
- Evaluation covers six downstream datasets:
  Browser, CSTNET-TLS, AppSniffer, ITC-Net-Blend, DataCon2021-p2, and AnonProxy.
- Reported baselines include supervised methods, content-only pre-training,
  behavior-aware pre-training, and multimodal baselines.
- The headline result is about +3.6 macro-F1 percentage points over the strongest
  per-task baseline on average, with larger gains under label scarcity.
- The ablation narrative is that Stage 1, CMC/ITC, CMM/ITM, masked
  reconstruction, both modalities, deep fusion, and ITGCA all contribute.
- The discussion admits two major limitations:
  inference cost and vulnerability to active behavior shaping.

When editing the paper, prefer aligning prose with the current code. In
particular, use the current ITGCA formula and implementation behavior, not older
draft notes.

## Follow-Up Research Direction

`followup_final.md` sketches a separate follow-up paper on robustness for
encrypted traffic classification. It should be treated as an independent
paper-level direction, not simply an MM-TrafficBERT extension.

The proposed follow-up positions MM-TrafficBERT as architecture work and the new
paper as a training wrapper that can sit around existing models.

Core follow-up idea:

- Separate traffic changes into:
  - A-type changes: actions that can be applied to a concrete flow, including
    natural network perturbations and adversarial evasion.
  - B-type changes: distribution drift/OOD caused by protocol evolution or
    long-term environment change.
- Defend A-type changes by using one protocol-realizable action menu. Natural
  changes are samples from that menu; adversarial changes are worst-case choices
  within that menu.
- Use B-type real drift as an evaluation axis, not as something the method claims
  to certify or fully defend.
- Proposed method direction includes a two-layer perturbation set:
  - Outer protocol budget box `B`, which is auditable and conservative.
  - Inner learned realizable manifold `M`, learned from real paired traffic or
    simulations and constrained within `B`.
- Candidate ingredients include natural sampling, adversarial worst-case search,
  differentiable proxies for non-differentiable traffic actions, channel balance
  regularization, and partial certification.

This follow-up direction connects naturally to the current paper's limitation on
behavior shaping. Current MM-TrafficBERT is strong on clean classification and
label efficiency, but all methods degrade under active size/IAT shaping. Future
work can target behavior-side reliability, shaping-aware pre-training, robust
fusion, and protocol-realizable adversarial training.

## Common Gotchas

- The local machine should not be used to run project workloads unless the user
  explicitly asks.
- Stage 2 pre-training and Stage 2 fine-tuning must use matching ITGCA and
  fusion geometry flags. Mismatches can discard gate weights or randomly
  initialize gate parameters.
- A Stage 2 checkpoint contains pre-training-only keys. Fine-tuning filters out
  momentum encoders, ITC projection heads, target heads, and queues.
- Stage 1 size/behavior checkpoints must include `temporal_embedding.weight`;
  otherwise the model was trained without IAT and performance will be invalid.
- The code writes intermediate `.pt` shards under `/tmp/<proc_id>.pt` during
  dataset building before merging.
- `followup_final.md` and `papers/followup_*.md` are research notes. They are not
  necessarily part of the current MM-TrafficBERT paper.
- `test/` contains useful experimental notes and figures but may include stale
  design drafts. Prefer current source code and `latex/ACM_conference/sections`
  when resolving conflicts.
- Large data/checkpoints are not stored directly in this repo. README references
  a Zenodo release for `pretrain.bin` and processed dataset splits.
