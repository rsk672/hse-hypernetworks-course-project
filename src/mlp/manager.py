import torch
from torch.optim import Adam
from torch import nn
import copy
from src.mlp.model import MLP
from src.hnet_lib.hnet import HyperNetwork
from utils.dataloaders import get_split_mnist_dataloaders


class MLPManager():
    def __init__(self, cmd_args):
        self.n_epochs = cmd_args.epochs
        self.hyper_layers = [int(c) - 1 for c in cmd_args.hypernet_layers]
        self.lr = cmd_args.learning_rate
        self.beta = cmd_args.beta
        self.batch_size = cmd_args.batch_size
        self.device = cmd_args.device
        self.use_random_embeddings = cmd_args.random_embeddings

        self.model = MLP(self.hyper_layers, self.device)
        self.hnet = HyperNetwork(
            5, 128, self.model.get_weights_from_hnet_cnt())
        self.optimizer = Adam(list(self.hnet.parameters()) +
                              list(self.model.parameters()), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.task_embeddings = []
        self.hnet_prev = None

        self.task_accuracies = []

    def train_task(self, dataloader):
        self.model.train()
        self.hnet.train()

        train_loss = 0
        train_acc = 0

        for i, batch in enumerate(dataloader):
            pixels, labels = batch
            pixels = pixels.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_random_embeddings:
                predicted_weights = self.hnet().to(self.device)
            else:
                predicted_weights = self.hnet.get_task_weights(
                    self.task_embeddings[-1]).to(self.device)

            output = self.model(pixels.view(
                pixels.shape[0], -1), predicted_weights)
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

        return train_loss / len(dataloader), train_acc / len(dataloader)

    def test_task(self, dataloader, task_embedding):
        self.model.eval()
        self.hnet.eval()

        test_loss = 0
        test_acc = 0

        for i, batch in enumerate(dataloader):
            pixels, labels = batch
            pixels = pixels.to(self.device)
            labels = labels.to(self.device)

            predicted_weights = self.hnet.get_task_weights(
                task_embedding).to(self.device)
            output = self.model(pixels.view(
                pixels.shape[0], -1), predicted_weights)

            loss = self.criterion(output, labels)

            test_loss += loss.item()
            test_acc += (output.argmax(dim=1) == labels).sum() / len(labels)

        return test_loss / len(dataloader), test_acc / len(dataloader)

    def train(self):
        train_tasks_loaders, test_tasks_loaders = get_split_mnist_dataloaders(
            self.batch_size)

        for t in range(5):
            print(f'TRAINING TASK {t + 1}\n')

            if not self.use_random_embeddings:
                current_task_embedding = torch.zeros(5)
                current_task_embedding[t] = 1
                self.task_embeddings.append(
                    current_task_embedding.to(self.device))

            for i in range(self.n_epochs):
                train_loss, train_acc = self.train_task(train_tasks_loaders[t])
                print(f'Epoch {i + 1}: {train_loss=} {train_acc=}')

            if t == 0:
                if len(self.hyper_layers) > 0:
                    self.model.freeze()
                else:
                    self.model.freeze_finetune()

            print(f'\nAFTER TRAINING TASK {t + 1}\n')

            accuracies = []
            if self.use_random_embeddings:
                self.task_embeddings.append(self.hnet.get_task_embedding())
            for i in range(t + 1):
                test_loss, test_acc = self.test_task(test_tasks_loaders[i],
                                                     self.task_embeddings[i])
                accuracies.append(test_acc.detach().item())
                print(f'Task {i + 1}: {test_loss=} {test_acc=}')
            self.task_accuracies.append(accuracies)

            self.hnet_prev = copy.deepcopy(self.hnet)
            self.hnet.reset_task_embedding()

            print()

    def get_task_accuracies(self):
        return self.task_accuracies
