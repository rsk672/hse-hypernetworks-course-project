import argparse
import numpy as np
from src.vit.custom.manager import CustomViTManager


def train_custom_vit(parsed_args):
    manager = CustomViTManager(cmd_args=parsed_args)
    manager.train()

    task_accuracies = manager.get_task_accuracies()

    task_accuracies_padded = [accs + [0] *
                              (5 - len(accs)) for accs in task_accuracies]

    output_name = 'custom-vit-hnet'

    if parsed_args.hyper_linear_probing:
        output_name += '-lp'
    if parsed_args.hyper_feedforward:
        output_name += '-ffd'
    if parsed_args.hyper_attention:
        output_name += '-attn'

    np.savetxt(
        f'experiments/custom-vit/accuracies/{output_name}', task_accuracies_padded, newline="\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Script to train custom ViT SplitCIFAR classifier using hypernetworks")

    args.add_argument(
        "-e",
        "--epochs",
        default=20,
        type=int,
        help="number of epochs to learn one task")

    args.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="learning rate for optimizer"
    )

    args.add_argument(
        "-b",
        "--beta",
        default=1e-2,
        type=float,
        help="regularization coefficient"
    )

    args.add_argument(
        "-bs",
        "--batch-size",
        default=64,
        type=int,
        help="batch size"
    )

    args.add_argument(
        "-d",
        "--device",
        default='cpu',
        type=str,
        help="cuda or cpu",
    )

    args.add_argument("-r", "--random-embeddings",
                      action=argparse.BooleanOptionalAction)

    args.add_argument("-hlp", "--hyper-linear-probing",
                      action=argparse.BooleanOptionalAction, help="if provided, then hypernetwork is used for the weights of last linear layer")

    args.add_argument("-hffd", "--hyper-feedforward",
                      action=argparse.BooleanOptionalAction, help="if provided, then hypernetwork is used for the weights of feed-forward layers")

    args.add_argument("-hattn", "--hyper-attention",
                      action=argparse.BooleanOptionalAction, help="if provided, then hypernetwork is used for the weights of attention layers")

    parsed_args = args.parse_args()

    train_custom_vit(parsed_args)
