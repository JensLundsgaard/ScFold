import copy
import os.path as osp
import pandas as pd

from API.cath_dataset import CATH
from API.ts_dataset import TS

from API.dataloader_gtrans import DataLoader_GTrans
from API.featurizer import featurize_GTrans
from API.h5_dataset import H5Dataset
from API.pdb_dataset import PDBDataset
import itertools

# I am editing this to insert my own Dataset into the dataloader
def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):
    # mdcath_spinet_320_0.h5 
    # mdcath_spinet_450_0.h5 
    # atlas_data.h5
    use_pdbs = True

    h5_path = osp.join("..", "atlas_data.h5")

    split_df = pd.read_csv(osp.join("..", "atlas_cross_val_index.csv"))
    val_mask = split_df["cross_val"] == 0 # change to whatever cross val sets you want
    test_mask = split_df["cross_val"] == 4

    train_indices = split_df[(~val_mask) & (~test_mask)]["random_indices"].to_list()
    val_indices = split_df[val_mask]["random_indices"].to_list()
    test_indices = split_df[test_mask]["random_indices"].to_list()
    
    val_groups = split_df[val_mask]["pdb"].to_list()
    train_groups = split_df[(~val_mask) & (~test_mask)]["pdb"].to_list()
    test_groups = split_df[test_mask]["pdb"].to_list()
    if not use_pdbs:
        train_set = H5Dataset(h5_path, random_indices=train_indices, groups=train_groups)
        valid_set = H5Dataset(h5_path, random_indices=val_indices, groups=val_groups)
        test_set = H5Dataset(h5_path, random_indices=test_indices, groups=test_groups)
    else:
        train_set = PDBDataset(train_groups)
        valid_set = PDBDataset(val_groups)
        test_set = PDBDataset(test_groups)

    collate_fn = featurize_GTrans
    print(type(collate_fn))
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
