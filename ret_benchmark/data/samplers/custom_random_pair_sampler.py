import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
# from torch.utils.data import Sampler
# import copy
# import numpy as np
# from collections import defaultdict
# import random

class CustomRandomPairSampler(Sampler):
    """
    Randomly sample pairs of images based on information from a CSV file.
    Args:
    - dataset (BaseDataSet).
    - batch_size (int): number of pairs in a batch.
    - csv_file (str): path to the CSV file containing pairs and labels.
    - max_iters (int): maximum number of iterations.
    """

    def __init__(self, dataset, batch_size, csv_file, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())
        self.csv_file = csv_file
        self.pairs = self._load_csv_data()

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Sampler | iters {self.max_iters} | M {self.batch_size} |"

    def _load_csv_data(self):
        pairs = []
        with open(self.csv_file, "r") as f:
            for line in f:
                img_path1, img_path2, comment = line.strip().split(",")
                pairs.append((img_path1, img_path2, comment))
        return pairs

    def _prepare_batch(self):
        batch_pairs = random.sample(self.pairs, self.batch_size)
        return batch_pairs

    def __iter__(self):
        for _ in range(self.max_iters):
            batch = self._prepare_batch()
            yield batch