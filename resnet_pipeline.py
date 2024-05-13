import argparse
import numpy as np
from src.resnet.manager import ResNetManager


def train_resnet(parsed_args):
    manager = ResNetManager(cmd_args=parsed_args)
    manager.train()

    task_accuracies = manager.get_task_accuracies()

    task_accuracies_padded = [accs + [0] *
                              (5 - len(accs)) for accs in task_accuracies]

    output_name = f'hnet-{parsed_args.hypernet_layers}' if parsed_args.hypernet_layers else f'ln'
    np.savetxt(
        f'experiments/resnet/accuracies/{output_name}', task_accuracies_padded, newline="\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Script to train ResNet18 SplitCIFAR classifier using hypernetworks")

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
        help="layers which weights are predicted by hypernetwork, should be a string of digits between 1 and 6. if not provided, then last layer fine-tuning will be used")

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

    args.add_argument("-hbn", "--hyper-batch-norm",
                      choices=["freezable", "full", "none"], default="none", help="""whether to use hyperbatchnorm layer or not. 
                      if 'freezable' is selected then running_mean and running_var are frozen after first task; 
                      if 'full' is selected then these parameters are changing while training; 
                      if 'none' is selected then the default batchnorm is used""")

    parsed_args = args.parse_args()

    train_resnet(parsed_args)
