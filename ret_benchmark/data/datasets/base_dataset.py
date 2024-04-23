import os
import re
import torch
from collections import defaultdict

from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image


# class BaseDataSet(Dataset):
#     """
#     Basic Dataset read image path from img_source
#     img_source: list of img_path and label
#     """

#     def __init__(self, img_source, transforms=None, mode="RGB"):
#         self.mode = mode
#         self.transforms = transforms
#         self.root = os.path.dirname(img_source)
#         assert os.path.exists(img_source), f"{img_source} NOT found."
#         self.img_source = img_source

#         self.label_list = list()
#         self.path_list = list()
#         self._load_data()
#         self.label_index_dict = self._build_label_index_dict()

#     def __len__(self):
#         return len(self.label_list)

#     def __repr__(self):
#         return self.__str__()

#     def __str__(self):
#         return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

#     def _load_data(self):
#         with open(self.img_source, "r") as f:
#             for line in f:
#                 _path, _label = re.split(r",", line.strip())
#                 self.path_list.append(_path)
#                 self.label_list.append(_label)

#     def _build_label_index_dict(self):
#         index_dict = defaultdict(list)
#         for i, label in enumerate(self.label_list):
#             index_dict[label].append(i)
#         return index_dict

#     def __getitem__(self, index):
#         path = self.path_list[index]
#         img_path = os.path.join(self.root, path)
#         label = self.label_list[index]

#         img = read_image(img_path, mode=self.mode)
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img, label, index


class BaseDataSet(Dataset):
    def __init__(self, img_folder, csv_file, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.img_folder = img_folder
        assert os.path.exists(img_folder), f"{img_folder} NOT found."
        self.csv_file = csv_file

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.csv_file, "r") as f:
            for line in f:
                _path1, _path2, _label = re.split(r",", line.strip())
                self.path_list.append((_path1, _path2))  # Store pairs of image paths
                self.label_list.append(int(_label))  # Convert label to integer

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path1, path2 = self.path_list[index]
        img_path1 = os.path.join(self.img_folder, path1)
        img_path2 = os.path.join(self.img_folder, path2)

        img1 = read_image(img_path1, mode=self.mode)
        img2 = read_image(img_path2, mode=self.mode)

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # For contrastive learning, assign label 1 if images are the same, 0 otherwise
        label = torch.tensor([int(self.label_list[index] == 1)], dtype=torch.float32)

        return img1, img2, label

