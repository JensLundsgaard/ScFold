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
    def __init__(self, h5_path, random_indices=None, groups=None):
        self.h5_path = h5_path
        self.h5_file = None
        self.groups = groups
        self.random_indices = random_indices # this says which frame we are taking from each traj, we'll just do shuffle=False and batch_size = 1000

        with h5py.File(self.h5_path) as f:
            self.build_groups(f)
            self.index = self.build_static_index(f)

    def __len__(self):
        return len(self.index)
    
    def get_item(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            
        group_idx, seq_idx = self.index[idx]
        group_name = self.groups[group_idx]

        coordinates = self.h5_file[group_name + "/coordinates"][:, seq_idx]
        y = "".join([seq1(res.decode()[:3]) for res in self.h5_file[group_name + "/residues"][:]])
        traj_id = group_name
        return {'title':traj_id, 'seq':y} | {atom: coordinates[:,i] for i, atom in enumerate(self.__class__.BACKBONE_ATOMS)}

    def __getitem__(self, idx):
        return self.get_item(idx)

    def build_groups(self, h5_file):
           use_split = self.groups is not None
           split = self.groups
           self.groups = []
           def visit(name, obj):
               if isinstance(obj, h5py.Group) and all(ds in obj for ds in self.__class__.REQUIRED_DATASETS) and (use_split <= (name.split("/")[0] in split)):
                   self.groups.append(name)
           h5_file.visititems(visit)

    def build_static_index(self, h5_file):
        index = []
        if self.random_indices is not None:
            for i, j in zip(range(len(self.groups)), self.random_indices):
                index.append((i, j))
        else:
            for i, group_name in enumerate(self.groups):
                for j in range(h5_file[group_name + "/" + "coordinates"].shape[1]):
                    index.append((i, j)) # otherwise we need to feed in whole sequences to do majority voting on
        return index

