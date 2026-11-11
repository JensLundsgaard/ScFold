import copy
import os.path as osp
import pandas as pd
import h5py
import numpy as np

from API.cath_dataset import CATH
from API.ts_dataset import TS
from torch.utils import data
from API.dataloader_gtrans import DataLoader_GTrans, featurize_GTrans
import inspect
from API.h5_dataset import H5Dataset
from API.pdb_dataset import PDBDataset
import itertools
import zlib
def pick_frame(num_frames, protein_id, seed, frame_index=None):
    if frame_index is not None:
        if not -num_frames <= frame_index < num_frames:
            raise IndexError("frame_index {} out of range for {} ({} frames)".format(
                frame_index, protein_id, num_frames))
        return int(frame_index % num_frames)
    rng = np.random.default_rng([int(seed), zlib.crc32(protein_id.encode())])
    return int(rng.integers(num_frames))

class AvgLogitsBatchSampler(data.BatchSampler):
    """
    return a bunch of frames from the same trajectory together
    """
    def __init__(self, raw_data, subsequence=128, step=32):
        super().__init__([], 1, False)
        list_names = [d["title"][0] for d in raw_data]
        self.names = list(set(list_names))
        self.index = list(enumerate(list_names))
        self.index = sorted(self.index, key=lambda tup: self.names.index(tup[1])) # sort names so it's in the right order and the for\while loop below is O(n)
        batches = []
        save_idx = 0
        for name in self.names:
            batch = []
            while(save_idx < len(self.index) and self.index[save_idx][1] == name): # this is now O(n) for n = len(raw_data)
                batch.append(self.index[save_idx][0])
                save_idx += 1
            if batch:
                batches.append(batch)
        self.subsequence = subsequence
        self.step = step
        self.batches = [[batch[i:i+self.subsequence] for i in range(0,len(batch)-self.subsequence+1, self.step)] for batch in batches]
        self.batches = [b for batch in self.batches for b in batch]# flatten

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

# I am editing this to insert my own Dataset into the dataloader
def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):
    if not isinstance(kwargs, dict):
        kwargs = vars(kwargs)
    h5_path = kwargs["h5_name"]
    index_path = kwargs["index_name"]
    use_pdbs = kwargs["use_pdbs"]
    test_val = kwargs["test_val"]

    h5_path = osp.join("..",h5_path)

    if(index_path == "atlas_cross_val_index.csv"):

        split_df = pd.read_csv(osp.join("..", index_path))
        val_mask = split_df["cross_val"] == 0 # change to whatever cross val sets you want
        test_mask = split_df["cross_val"] == 4

        #train_indices = split_df[(~val_mask) & (~test_mask)]["random_indices"].to_list()
        #val_indices = split_df[val_mask]["random_indices"].to_list()
        #test_indices = split_df[test_mask]["random_indices"].to_list()
        
        val_groups = split_df[val_mask]["pdb"].to_list()
        train_groups = split_df[(~val_mask) & (~test_mask)]["pdb"].to_list()
        test_groups = split_df[test_mask]["pdb"].to_list()


    else:

        index = pd.read_csv(osp.join("..", index_path))

        train_groups = index[index["split"] == "train"]["domain"].tolist()
        test_groups = index[index["split"] == "test"]["domain"].tolist()
        val_groups = index[index["split"] == "validation"]["domain"].tolist()        

    """

    pdb_to_size = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "coordinates" in obj:
            pdb_to_size[name.split("/")[0]] = obj["coordinates"].shape[1]
    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)

    train_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in train_groups]
    val_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in val_groups]
    test_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in test_groups]

    """

    train_set = H5Dataset(h5_path, groups=train_groups, step=16)
    valid_set = H5Dataset(h5_path, groups=val_groups, step=1)
    test_set = H5Dataset(h5_path, groups=test_groups, step=1)
    
    """
    train_set = PDBDataset(train_groups, h5_path, num_workers=16, verbose=True)
    valid_set = PDBDataset(val_groups, h5_path, num_workers=16, verbose=True)
    test_set = PDBDataset(test_groups, h5_path, num_workers=16, verbose=True)
    """

    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))

    print(len(train_groups))
    print(len(val_groups))
    print(len(test_groups))

    collate_fn = featurize_GTrans

    val_sampler = AvgLogitsBatchSampler(valid_set.samples)
    test_sampler = AvgLogitsBatchSampler(test_set.samples)

    train_loader = DataLoader_GTrans(train_set, batch_size=16, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader_GTrans(valid_set, batch_sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader_GTrans(test_set, batch_sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn)

    # test
    assert len(next(iter(train_loader))) == 6, "train_loader does not output 6 values"

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn)

    return test_loader
