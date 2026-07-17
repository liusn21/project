import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam, apply_modality_configs
from uer.opts import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--vocab_path_temporal", default=None, type=str,
                        help="Path of the temporal vocabulary file (for IAT tokens in packet_size target).")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json", #define the model
                        help="Config file of model hyper-parameters.")
    parser.add_argument("--config_path_raw", type=str, default=None,
                        help="Optional per-modality config for the Raw (content) encoder. "
                             "Only 'layers_num' is applied; shared dims (hidden_size/heads_num/emb_size) "
                             "must match --config_path. Falls back to --config_path if unset.")
    parser.add_argument("--config_path_size", type=str, default=None,
                        help="Optional per-modality config for the Size (behavior) encoder. "
                             "Same rules as --config_path_raw.")

    # Training and saving options. 
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-process micro-batch size. Optimizer global batch is "
                             "batch_size x world_size x accumulation_steps; Stage-2 "
                             "ITC/ITM candidate sets remain micro-batch-local.")
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert"], default="bert",
                        help="Specify the tokenizer.")

    # Model options.
    model_opts(parser)
    parser.add_argument("--target", choices=["raw_packet", "packet_size", "multimodal"], default="multimodal",
                        help="The training target of the pretraining model.")

    # Masking options.
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    # Optimizer options.
    optimization_opts(parser)

    # Multi-Modal options (Stage 2 - ALBEF-style).
    parser.add_argument("--seq_length_raw", type=int, default=512,
                        help="Sequence length for Raw Packet modality.")
    parser.add_argument("--seq_length_size", type=int, default=256,
                        help="Sequence length for Packet Size modality.")
    parser.add_argument("--vocab_path_raw", default=None, type=str,
                        help="Path of the vocabulary file for Raw Packet modality.")
    parser.add_argument("--vocab_path_size", default=None, type=str,
                        help="Path of the vocabulary file for Packet Size modality.")
    parser.add_argument("--corpus_path_raw", default=None, type=str,
                        help="Path of the corpus file for Raw Packet modality.")
    parser.add_argument("--corpus_path_size", default=None, type=str,
                        help="Path of the corpus file for Packet Size modality.")
    parser.add_argument("--pretrained_raw_path", default=None, type=str,
                        help="Path of the pretrained Raw Packet encoder (from Stage 1).")
    parser.add_argument("--pretrained_size_path", default=None, type=str,
                        help="Path of the pretrained Packet Size encoder (from Stage 1).")

    # ALBEF-style architecture parameters
    parser.add_argument("--num_fusion_layers", type=int, default=6,
                        help="Number of bidirectional cross-attention fusion layers.")
    parser.add_argument("--queue_size", type=int, default=4096,
                        help="Size of the feature queue for ITC contrastive learning.")
    parser.add_argument("--momentum", type=float, default=0.995,
                        help="Momentum coefficient for EMA update of momentum encoders.")

    # Loss weights
    parser.add_argument("--lambda_itc", type=float, default=1.0,
                        help="Weight for ITC (contrastive) loss.")
    parser.add_argument("--lambda_itm", type=float, default=1.0,
                        help="Weight for ITM (matching) loss.")
    parser.add_argument("--lambda_mlm", type=float, default=1.0,
                        help="Weight for MLM losses (both raw and size).")
    parser.add_argument("--lambda_mlm_raw", type=float, default=None,
                        help="Weight for Raw MLM loss (if set, overrides lambda_mlm for raw).")
    parser.add_argument("--lambda_mlm_size", type=float, default=1.0,
                        help="Weight for Size MLM loss (for packet_size target with temporal).")
    parser.add_argument("--lambda_mlm_temporal", type=float, default=1.0,
                        help="Weight for Temporal MLM loss (for packet_size target with temporal IAT).")
    parser.add_argument("--lambda_recon_size", type=float, default=1.0,
                        help="Weight for Size reconstruction loss (for multimodal target).")
    parser.add_argument("--lambda_recon_temporal", type=float, default=1.0,
                        help="Weight for Temporal reconstruction loss (for multimodal target).")

    # ITGCA (Information-Theoretic Gated Cross-Attention) options
    parser.add_argument("--use_itgca", action="store_true",
                        help="Enable asymmetric ITGCA gate mechanism in fusion layers. "
                             "Size←Raw: source-side V gating + modality prior + learned token gate. "
                             "Raw←Size: pure learned gate (no prior).")
    parser.add_argument("--itgca_window_size", type=int, default=16,
                        help="Sliding window size for local entropy computation in ITGCA token gate. "
                             "Recommended range: 16-32 for byte-level token sequences.")

    # ITGCA component-level ablation flags (for §5.5.2 component decomposition)
    # Defaults: all False = full ITGCA. Each flag disables one sub-component cleanly.
    parser.add_argument("--ablate_r_stat", action="store_true",
                        help="Disable the flow-level Shannon entropy prior r_stat. "
                             "Gate falls back to r_mod = r_learned on Size←Raw direction.")
    parser.add_argument("--ablate_g_token", action="store_true",
                        help="Disable the token-level gate g_token (forced to 1.0 constant). "
                             "Hierarchical gating degrades to flow-level only.")
    parser.add_argument("--ablate_source_bias", action="store_true",
                        help="Disable the source-side attention bias b_j computed from "
                             "sliding-window local entropy. Affects Size←Raw direction only.")

    # ITGCA initialization values (for §5.3.4 sensitivity-to-initialization experiments)
    parser.add_argument("--alpha_init", type=float, default=-2.0,
                        help="Initial value for alpha_modality (gated residual mixing). "
                             "sigmoid(alpha_init) controls initial blend between r_stat and r_learned. "
                             "Default -2.0 → sigmoid ≈ 0.12 (heavy prior reliance).")
    parser.add_argument("--token_gate_bias_init", type=float, default=2.0,
                        help="Initial value for token_gate_bias. "
                             "sigmoid(token_gate_bias_init) controls default gate openness. "
                             "Default 2.0 → sigmoid ≈ 0.88 (default-open).")

    # Temperature parameters
    parser.add_argument("--itc_temperature", type=float, default=0.07,
                        help="Temperature for ITC contrastive loss.")
    parser.add_argument("--itm_temperature", type=float, default=0.07,
                        help="Temperature for ITM hard negative sampling.")

    # ITC momentum distillation (ALBEF). alpha=0 disables it (pure hard-label
    # InfoNCE); alpha>0 mixes in soft targets from the momentum model. The value
    # is linearly ramped from 0 to itc_alpha over the LR-warmup window.
    parser.add_argument("--itc_alpha", type=float, default=0.4,
                        help="Max weight for ITC momentum-distillation soft targets "
                             "(ALBEF). 0 = disabled (hard-label InfoNCE only).")

    # Temporal soft-label parameters
    parser.add_argument("--temporal_sigma", type=float, default=10,
                        help="Standard deviation for temporal soft-label KL loss. "
                             "Controls tolerance range: ±3σ covers ~99.7% probability mass. "
                             "Default=10 means predictions within ±30 tokens are considered 'close'.")

    # Learning rate
    parser.add_argument("--encoder_lr_ratio", type=float, default=0.1,
                        help="Learning rate ratio for encoders. "
                             "E.g., 0.1 means encoder LR = base_LR * 0.1")

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")


    args = parser.parse_args()
    print("Training arguments:")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Encoder learning rate ratio: {args.encoder_lr_ratio}")

    # Load hyper-parameters from config file.
    if args.config_path:
        load_hyperparam(args)
    apply_modality_configs(args)

    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        # Multiprocessing distributed mode.
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        # Single GPU mode.
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        print("args.gpu_id:",args.gpu_id)
        print("torch.cuda.device_count,",torch.cuda.device_count())
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        print("Using GPU %d for training." % args.gpu_id)
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()
