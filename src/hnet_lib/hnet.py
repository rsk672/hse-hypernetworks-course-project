import torch
from torch import nn

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device='cpu'):
        super().__init__()

        self.hypernetwork_input = nn.Parameter(
            torch.randn((1, input_dim)), requires_grad=True).float()

        self.hnet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.device = device

    def reset_task_embedding(self):
        self.hypernetwork_input = nn.Parameter(
            torch.randn((1, self.hypernetwork_input.shape[1])), requires_grad=True).float()

    def get_task_embedding(self):
        return self.hypernetwork_input.data.detach().clone().squeeze(0)

    def get_task_weights(self, t):
        return self.hnet(t.to(self.device))

    def forward(self):
        return self.hnet(self.hypernetwork_input.to(self.device)).squeeze(0)