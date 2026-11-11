"""
Author: Jens Lundsgaard
Based on: cath_dataset.py
"""
import os
import json
import traceback

import numpy as np
from tqdm import tqdm

import torch.utils.data as data
import h5py
import torch
from Bio.SeqUtils import seq1

class H5Dataset(data.Dataset):
    BACKBONE_ATOMS = ["CA", "N", "C", "O"]
    REQUIRED_DATASETS = {"coordinates", "dihedrals", "spinet_features","frame_maps","residues"}
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU","LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

    def __init__(self, h5_path, random_indices=None, groups=None):
        self.h5_path = h5_path
        self.h5_file = None
        self.groups = groups
        self.random_indices = random_indices
        self.step = 16

        self.samples = []
        use_split = self.groups is not None
        split = self.groups

        def visit(name, obj):
            group_name = name.split("/")[0]
            if isinstance(obj, h5py.Group) and all(ds in obj for ds in self.__class__.REQUIRED_DATASETS) and (use_split <= (group_name in split)):
                coordinates = obj["coordinates"][:, ::self.step]
                y = "".join([seq1(self.__class__.RESIDUE_ALIASES.get(res.decode()[:3], res.decode()[:3])) for res in obj["residues"][:]])
                assert all((y_char in self.__class__.ALPHABET) for y_char in y), f"y contains: {[y_char for y_char in y if (y_char not in self.__class__.ALPHABET)][0]} which is not in {self.__class__.ALPHABET}"
                for idx in range(coordinates.shape[1]):
                    self.samples.append({'title':(group_name, idx * self.step), 'seq':y} | {atom: coordinates[:,idx,i] for i, atom in enumerate(self.__class__.BACKBONE_ATOMS)})


        with h5py.File(self.h5_path) as f:
            f.visititems(visit)

            
               

    def __len__(self):
        return len(self.index)
    
    def get_item(self, idx):
        return self.samples[idx]

    def __getitem__(self, idx):
        return self.get_item(idx)

    def build_static_index(self, h5_file):
        index = []
        if self.random_indices is not None:
            for i, j in zip(range(len(self.groups)), self.random_indices):
                index.append((i, j))
        else:
            for i, group_name in enumerate(self.groups):
                for j in range(0, h5_file[group_name]["coordinates"].shape[1] if self.max_len is None else self.max_len, self.step):
                    index.append((i, j)) 
        return index

