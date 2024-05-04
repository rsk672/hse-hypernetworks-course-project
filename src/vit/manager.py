import torch
from torch import nn
from torch.optim import Adam
import copy
from src.hnet_lib.hnet import HyperNetwork
from src.vit.model_pretrained import PreTrainedViT
from utils.dataloaders import get_split_cifar10_dataloaders, get_split_cifar100_dataloaders


class PreTrainedViTManager():
    def __init__(self, cmd_args):
        super().__init__()

        self.n_epochs = cmd_args.epochs
        self.lr = cmd_args.learning_rate
        self.beta = cmd_args.beta
        self.batch_size = cmd_args.batch_size
        self.device = cmd_args.device
        self.use_random_embeddings = cmd_args.random_embeddings
        self.dataset = cmd_args.dataset
        self.num_classes = 10 if self.dataset == 'cifar10' else 100
        self.tasks_count = 5 if self.dataset == 'cifar10' else 10

        self.model = PreTrainedViT(
            self.num_classes, use_hnet=cmd_args.use_hypernetwork)
        self.hnet = HyperNetwork(self.tasks_count, 128, self.model.get_weights_from_hnet_cnt(
        )) if cmd_args.use_hypernetwork else None
        self.criterion = nn.CrossEntropyLoss()

        if self.hnet:
            self.optimizer = Adam(list(self.hnet.parameters()) +
                                  list(self.model.parameters()), lr=self.lr)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.task_embeddings = []
        self.hnet_prev = None
        self.task_accuracies = []

    def train_task(self, dataloader):
        self.model.train()

        if self.hnet:
            self.hnet.train()

        train_loss = 0
        train_acc = 0

        for i, batch in enumerate(dataloader):
            pixels, labels = batch
            pixels = pixels.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predicted_weights = None

            if self.hnet:
                if self.use_random_embeddings:
                    current_task_embedding = self.hnet().to(self.device)
                    predicted_weights = self.hnet().to(self.device)
                else:
                    current_task_embedding = self.task_embeddings[-1]
                    predicted_weights = self.hnet.get_task_weights(
                        current_task_embedding).to(self.device)

            output = self.model(pixels, predicted_weights)

            loss = self.criterion(output, labels)

            if self.hnet_prev:
                prev_task_embeddings = self.task_embeddings if self.use_random_embeddings else self.task_embeddings[
                    :-1]
                for t_e in prev_task_embeddings:
                    loss += (self.beta / len(prev_task_embeddings)) * torch.sum(
                        (self.hnet_prev.get_task_weights(t_e) - self.hnet.get_task_weights(t_e)) ** 2)

            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == labels).sum() / len(labels)
            loss.backward()

            self.optimizer.step()

            break

        return train_loss / len(dataloader), train_acc / len(dataloader)

    def test_task(self, dataloader, task_embedding=None):
        self.model.eval()

        if self.hnet:
            self.hnet.eval()

        test_loss = 0
        test_acc = 0

        for i, batch in enumerate(dataloader):
            pixels, labels = batch
            pixels = pixels.to(self.device)
            labels = labels.to(self.device)

            predicted_weights = None

            if self.hnet:
                predicted_weights = self.hnet.get_task_weights(
                    task_embedding).to(self.device)

            output = self.model(pixels, predicted_weights)

            loss = self.criterion(output, labels)

            test_loss += loss.item()
            test_acc += (output.argmax(dim=1) == labels).sum() / len(labels)

            break

        return test_loss / len(dataloader), test_acc / len(dataloader)

    def train(self):
        if self.dataset == 'cifar10':
            train_tasks_loaders, test_tasks_loaders = get_split_cifar10_dataloaders(
                self.batch_size, 224)
        else:
            train_tasks_loaders, test_tasks_loaders = get_split_cifar100_dataloaders(
                self.batch_size, 224)

        for t in range(self.tasks_count):
            print(f'TRAINING TASK {t + 1}\n')

            if self.hnet and not self.use_random_embeddings:
                current_task_embedding = torch.zeros(self.tasks_count)
                current_task_embedding[t] = 1
                self.task_embeddings.append(
                    current_task_embedding.to(self.device))

            for i in range(self.n_epochs):
                train_loss, train_acc = self.train_task(train_tasks_loaders[t])
                print(f'Epoch {i + 1}: {train_loss=} {train_acc=}')

            print(f'\nAFTER TRAINING TASK {t + 1}\n')

            accuracies = []
            if self.hnet and self.use_random_embeddings:
                self.task_embeddings.append(self.hnet.get_task_embedding())

            for i in range(t + 1):
                task_embedding = None
                if self.hnet:
                    task_embedding = self.task_embeddings[i]

                test_loss, test_acc = self.test_task(test_tasks_loaders[i],
                                                     task_embedding)
                accuracies.append(test_acc.detach().item())
                print(f'Task {i + 1}: {test_loss=} {test_acc=}')

            self.task_accuracies.append(accuracies)

            if self.hnet:
                self.hnet_prev = copy.deepcopy(self.hnet)
                self.hnet.reset_task_embedding()

            print()

    def get_task_accuracies(self):
        return self.task_accuracies
