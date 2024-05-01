import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_tasks_bar(task_accuracies_path, output_path, title):
    task_accuracies = np.loadtxt(task_accuracies_path).T

    task_labels = ("Learned Task 1", "Learned Task 2",
                   "Learned Task 3", "Learned Task 4", "Learned Task 5")

    tasks = {
        '0-1': task_accuracies[0],
        '2-3': task_accuracies[1],
        '4-5': task_accuracies[2],
        '6-7': task_accuracies[3],
        '8-9': task_accuracies[4],
    }

    x = np.arange(1, len(task_labels) + 1)
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in tasks.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Tasks learned')
    if title is not None:
        ax.set_title(f'{title}')
    ax.legend(loc='upper right', ncols=5)
    ax.set_ylim(0, 1.2)

    ax.locator_params(axis='y', nbins=20)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-2].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description="Script to plot task accuracies")

    args.add_argument(
        "-i",
        "--input-path",
        type=str,
        help="task accuracies path",
        required=True,
    )

    args.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="path where barchart will be saved",
        required=True,
    )

    args.add_argument(
        "-t",
        "--title",
        type=str,
        help="barchart title"
    )

    parsed_args = args.parse_args()

    plot_tasks_bar(parsed_args.input_path,
                   parsed_args.output_path, parsed_args.title)
