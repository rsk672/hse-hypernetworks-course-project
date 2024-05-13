import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_tasks_bar(task_accuracies_path, output_path, title=None, full_training_accuracy=None):
    task_accuracies = np.loadtxt(task_accuracies_path).T * 100

    tasks = {
        '0-1': task_accuracies[0],
        '2-3': task_accuracies[1],
        '4-5': task_accuracies[2],
        '6-7': task_accuracies[3],
        '8-9': task_accuracies[4],
    }

    x = np.arange(1, len(tasks) + 1)
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    handles = []

    for attribute, measurement in tasks.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        handles.append(rects)
        multiplier += 1

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Выучено контекстов')
    if title is not None:
        ax.set_title(f'{title}')


    if full_training_accuracy is not None:
      hline = ax.hlines(full_training_accuracy, 0.92, 5.88, color='black',
                        linestyle='--', linewidth=2, label='Full Training Accuracy')
      handles.append(hline)

    ax.legend(handles=handles, loc='upper right', ncols=5, fontsize=12)
    ax.set_ylim(0, 120)

    ax.locator_params(axis='y', nbins=20)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-2].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-2].label1.set_visible(False)
    yticks[-1].label1.set_visible(False)

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    ax.tick_params(axis='both', which='major', labelsize=16)

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

    args.add_argument(
        "-f",
        "--full-training-accuracy",
        type=float,
        help="accuracy of model when trained on full dataset (not continually)"
    )

    parsed_args = args.parse_args()

    plot_tasks_bar(parsed_args.input_path,
                   parsed_args.output_path, parsed_args.title, parsed_args.full_training_accuracy)
