import copy
import os.path as osp
import pandas as pd
import h5py
import numpy as np

from API.cath_dataset import CATH
from API.ts_dataset import TS

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

# I am editing this to insert my own Dataset into the dataloader
def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):
    if not isinstance(kwargs, dict):
        kwargs = vars(kwargs)
    #h5_path = kwargs["h5_name"]
    index_path = kwargs["index_name"]
    use_pdbs = kwargs["use_pdbs"]
    test_val = kwargs["test_val"]

    if(index_path == "atlas_cross_val_index.csv"):

        #h5_path = osp.join("..",h5_path)
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

        #h5_path = osp.join("..",h5_path)
        index = pd.read_csv(osp.join("..", index_path))

        train_groups = index[index["split"] == "train"]["domain"].tolist()
        test_groups = index[index["split"] == "test"]["domain"].tolist()
        val_groups = index[index["split"] == "validation"]["domain"].tolist()        
        #pdb_to_size = {}
        #def visit(name, obj):
        #    if isinstance(obj, h5py.Group) and "coordinates" in obj:
        #        pdb_to_size[name.split("/")[0]] = obj["coordinates"].shape[1]
        #with h5py.File(h5_path, "r") as f:
        #    f.visititems(visit)

        #train_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in train_groups]
        #val_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in val_groups]
        #test_indices = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in test_groups]


        #train_set = H5Dataset(h5_path, random_indices=train_indices, groups=train_groups)
        #valid_set = H5Dataset(h5_path, random_indices=val_indices, groups=val_groups)
        #test_set = H5Dataset(h5_path, random_indices=test_indices, groups=test_groups)
    #else:

    train_set = PDBDataset(train_groups, use_cath=(index_path != 'atlas_cross_val_index.csv'))
    valid_set = PDBDataset(val_groups, use_cath=(index_path != 'atlas_cross_val_index.csv'))
    test_set = PDBDataset(test_groups, use_cath=(index_path != 'atlas_cross_val_index.csv'))

    collate_fn = featurize_GTrans

    train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    # test
    assert len(next(iter(train_loader))) == 6, "train_loader does not output 6 values"

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn)

    return test_loader
