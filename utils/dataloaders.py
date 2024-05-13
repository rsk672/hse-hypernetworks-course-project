from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from utils.task_splitter import TaskSplitter


def get_split_mnist_dataloaders(batch_size):
    mnist_mean = 0.1307
    mnist_std = 0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])

    mnist_train_dataset = datasets.MNIST('src/data/mnist', train=True,
                                         download=True, transform=transform)
    mnist_test_dataset = datasets.MNIST(
        'src/mlp/mnist/', download=True, train=False, transform=transform)

    train_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(mnist_train_dataset):
        train_target_indices[target].append(i)

    test_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(mnist_test_dataset):
        test_target_indices[target].append(i)

    label_pairs = [[i, i + 1] for i in range(0, 9, 2)]
    train_tasks_loaders = []
    test_tasks_loaders = []

    for pair in label_pairs:
        train_task = TaskSplitter(full_dataset=mnist_train_dataset,
                                  target_indices_dict=train_target_indices, task_targets=pair)
        test_task = TaskSplitter(full_dataset=mnist_test_dataset,
                                 target_indices_dict=test_target_indices, task_targets=pair)

        train_tasks_loaders.append(DataLoader(
            train_task, batch_size=batch_size))
        test_tasks_loaders.append(DataLoader(test_task, batch_size=batch_size))

    return train_tasks_loaders, test_tasks_loaders


def get_split_cifar10_dataloaders(batch_size, image_size=32, use_augmentations=False):

    if not use_augmentations:
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    cifar10_train_dataset = datasets.CIFAR10(root='src/data/cifar10', train=True,
                                             download=True, transform=transform)

    cifar10_test_dataset = datasets.CIFAR10(root='src/data/cifar10', train=False,
                                            download=True, transform=transform)

    train_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(cifar10_train_dataset):
        train_target_indices[target].append(i)

    test_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(cifar10_test_dataset):
        test_target_indices[target].append(i)

    label_pairs = [[i, i + 1] for i in range(0, 9, 2)]
    train_tasks_loaders = []
    test_tasks_loaders = []

    for pair in label_pairs:
        train_task = TaskSplitter(full_dataset=cifar10_train_dataset,
                                  target_indices_dict=train_target_indices, task_targets=pair)
        test_task = TaskSplitter(full_dataset=cifar10_test_dataset,
                                 target_indices_dict=test_target_indices, task_targets=pair)

        train_tasks_loaders.append(DataLoader(
            train_task, batch_size=batch_size))
        test_tasks_loaders.append(DataLoader(test_task, batch_size=batch_size))

    return train_tasks_loaders, test_tasks_loaders


def get_split_cifar100_dataloaders(batch_size, image_size=32):

    transform = transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar100_train_dataset = datasets.CIFAR100(root='src/data/cifar100', train=True,
                                              download=True, transform=transform)

    cifar100_test_dataset = datasets.CIFAR100(root='src/data/cifar100', train=False,
                                             download=True, transform=transform)

    train_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(cifar100_train_dataset):
        train_target_indices[target].append(i)

    test_target_indices = defaultdict(list)
    for i, (pixels, target) in enumerate(cifar100_test_dataset):
        test_target_indices[target].append(i)

    label_group = [[i for i in range(i, i + 10)] for i in range(0, 100, 10)]
    train_tasks_loaders = []
    test_tasks_loaders = []

    for group in label_group:
        train_task = TaskSplitter(full_dataset=cifar100_train_dataset,
                                  target_indices_dict=train_target_indices, task_targets=group)
        test_task = TaskSplitter(full_dataset=cifar100_test_dataset,
                                 target_indices_dict=test_target_indices, task_targets=group)

        train_tasks_loaders.append(DataLoader(
            train_task, batch_size=batch_size))
        test_tasks_loaders.append(DataLoader(test_task, batch_size=batch_size))

    return train_tasks_loaders, test_tasks_loaders
