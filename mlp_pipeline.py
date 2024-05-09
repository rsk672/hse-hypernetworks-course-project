import argparse
import numpy as np
from src.mlp.manager import MLPManager


def train_mlp(parsed_args):
    manager = MLPManager(cmd_args=parsed_args)
    manager.train()

    task_accuracies = manager.get_task_accuracies()

    task_accuracies_padded = [accs + [0] *
                              (5 - len(accs)) for accs in task_accuracies]
    
    output_name = f'hnet-{parsed_args.hypernet_layers}' if parsed_args.hypernet_layers else f'fine-tuning.txt'
    np.savetxt(
        f'experiments/mlp/accuracies/{output_name}', task_accuracies_padded, newline="\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Script to train MLP SplitMNIST classifier using hypernetworks")

    args.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="number of epochs to learn one task")

    args.add_argument(
        "-hl",
        "--hypernet-layers",
        default="",
        type=str,
        help="layers which weights are predicted by hypernetwork, should be a string of digits between 0 and 2. if not provided, then vanilla fine-tuning will be used")

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

    parsed_args = args.parse_args()

    train_mlp(parsed_args)
