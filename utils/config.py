import argparse


TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSY_VALUES = {"0", "false", "f", "no", "n", "off"}


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate HRCH.")

    parser.add_argument("--data_name", type=str, default="iapr", help="Dataset name.")
    parser.add_argument("--data_root", type=str, default="", help="Root directory of the selected dataset.")
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="",
        help="Path to the RevCol pretrained backbone weights.",
    )
    parser.add_argument("--root_dir", type=str, default=".", help="Repository root used for outputs.")
    parser.add_argument("--log_name", type=str, default="HRCH", help="TensorBoard log directory name.")
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default="HRCH",
        help="Subdirectory name used for checkpoints and exported features.",
    )
    parser.add_argument("--arch", type=str, default="revc", help="Backbone name used in checkpoint filenames.")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--log_interval", type=int, default=40, help="Logging interval in iterations.")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of dataloader workers.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer type.")
    parser.add_argument("--ls", type=str, default="linear", choices=["linear", "cos"], help="LR scheduler type.")
    parser.add_argument("--warmup_epoch", type=int, default=0, help="Number of warmup epochs.")
    parser.add_argument(
        "--train_eval_query_size",
        type=int,
        default=2000,
        help="Number of train-side retrieval samples used to monitor avg during training. Use 0 for the full retrieval split.",
    )
    parser.add_argument(
        "--early_stop_decline_patience",
        type=int,
        default=4,
        help="Stop training if the monitored avg declines for this many consecutive epochs.",
    )
    parser.add_argument(
        "--early_stop_best_patience",
        type=int,
        default=8,
        help="Stop training if the monitored avg does not beat the best value for this many consecutive epochs.",
    )

    parser.add_argument("--bit", type=int, default=32, help="Hash code length.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Weight for clustering loss.")
    parser.add_argument("--momentum", type=float, default=0.4, help="Momentum used by legacy NCE memory.")
    parser.add_argument("--K", type=int, default=4096, help="Legacy NCE queue size.")
    parser.add_argument("--T", type=float, default=0.9, help="Legacy NCE temperature.")
    parser.add_argument("--shift", type=float, default=1.0, help="Contrastive shift.")
    parser.add_argument("--margin", type=float, default=0.2, help="Contrastive margin.")

    parser.add_argument("--tau", type=float, default=0.12, help="Instance-level clustering temperature.")
    parser.add_argument("--taup", type=float, default=0.12, help="Prototype-level clustering temperature.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--ins", type=float, default=0.4, help="Instance clustering loss weight.")
    parser.add_argument("--pro", type=float, default=1.6, help="Prototype clustering loss weight.")
    parser.add_argument("--cluster_num", type=str, default="1000,500,100,50", help="Comma-separated cluster sizes.")
    parser.add_argument("--layers", type=str, default="2,2,4,2", help="Comma-separated image backbone layers.")
    parser.add_argument("--ld", type=int, default=4, help="Legacy layer decay parameter kept for compatibility.")
    parser.add_argument("--dp", type=float, default=0.01, help="Drop path rate.")
    parser.add_argument("--droprate", type=float, default=0.0, help="Dropout rate used by hash heads.")
    parser.add_argument("--entroy", type=float, default=0.1, help="Entropy regularization weight.")
    parser.add_argument("--qua", type=float, default=0.01, help="Quantization regularization weight.")

    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--pseed",
        type=int,
        default=3407,
        help="Prototype sampling seed used by the original IAPR experiments.",
    )
    parser.add_argument(
        "--feature_save",
        type=str2bool,
        default=False,
        help="Whether to export retrieval features after loading a checkpoint.",
    )
    parser.add_argument(
        "--eval_only",
        type=str2bool,
        default=False,
        help="Skip training and evaluate a checkpoint only.",
    )
    parser.add_argument(
        "--res",
        type=str2bool,
        default=False,
        help="Deprecated alias for --eval_only kept for backward compatibility.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path used for resume, evaluation or feature export.",
    )

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    args.eval_only = args.eval_only or args.res
    return args
