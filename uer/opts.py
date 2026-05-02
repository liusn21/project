def model_opts(parser):
    parser.add_argument("--embedding", choices=["raw_packet", "packet_size", "multimodal"],
                        default="raw_packet",
                        help="Embedding type (kept for backward compat with --embedding flag).")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--encoder", choices=["transformer"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--mask", choices=["fully_visible"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) native PyTorch AMP precision.")
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts"],
                        default="linear", help="Scheduler type.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping. Set to 0 to disable.")


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")


def finetune_opts(parser):
    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str,
                        help="Path of the devset.")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Optimization options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)
