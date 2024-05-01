import torch
from torch.utils.data import Dataset


class TaskSplitter(Dataset):
    def __init__(self, full_dataset, target_indices_dict, task_targets):
        self._full_dataset = full_dataset

        self._task_indices = []
        for target in task_targets:
            self._task_indices += target_indices_dict[target]
        self._task_indices = torch.tensor(
            sorted(self._task_indices), dtype=torch.int32)

    def __len__(self):
        return len(self._task_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._full_dataset[self._task_indices[idx]]
