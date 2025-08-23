from torch.utils import data
from torch_geometric.data import Data

from utils.utils import *


class MIMICBaseDataset(data.Dataset):

    def __init__(self,
                 patients: dict,
                 task: str,
                 graph: Data,
                 prescriptions_code_to_name: dict,
                 phenotype_names: list):

        self.task = task
        self.graph = graph
        self.prescriptions_code_to_name = prescriptions_code_to_name
        self.phenotype_names = phenotype_names
        self.dataset = self.extract_data(patients)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def extract_data(self, patients: dict):

        dataset = []

        for patient in patients.values():
            dataset.append(patient)

        return dataset
