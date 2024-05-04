import argparse
import numpy as np
from src.vit.manager import PreTrainedViTManager


def pretrained_vit_linear_probing(parsed_args):
    manager = PreTrainedViTManager(cmd_args=parsed_args)
    manager.train()

    task_accuracies = manager.get_task_accuracies()

    task_accuracies_padded = [accs + [0] *
                              (len(task_accuracies[-1]) - len(accs)) for accs in task_accuracies]

    output_name = 'pretrained-vip-hnet-lp' if parsed_args.use_hypernetwork else 'pretrained-vip-lp'

    if parsed_args.dataset == "cifar10":
        np.savetxt(
            f'experiments/split-cifar10/accuracies/{output_name}', task_accuracies_padded, newline="\n")
    else:
        np.savetxt(
            f'experiments/split-cifar100/accuracies/{output_name}', task_accuracies_padded, newline="\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Script to train pre-trained ViT model SplitCIFAR classifier using hypernetworks")

    args.add_argument(
        "-e",
        "--epochs",
        default=2,
        type=int,
        help="number of epochs to learn one task")

    args.add_argument("-uh", "--use-hypernetwork",
                      action=argparse.BooleanOptionalAction, help="whether to use hypernetwork for linear probing")

    args.add_argument(
        "-ds", "--dataset", choices=["cifar10", "cifar100"], default="cifar10", help="which dataset will be used for continual learning")

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
                      action=argparse.BooleanOptionalAction, help="if this option is provided then task embeddings are learned during training")

    parsed_args = args.parse_args()

    pretrained_vit_linear_probing(parsed_args)
