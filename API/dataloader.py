import copy
import os.path as osp
import pandas as pd

from API.cath_dataset import CATH
from API.ts_dataset import TS

from API.dataloader_gtrans import DataLoader_GTrans
from API.featurizer import featurize_GTrans
from API.h5_dataset import H5Dataset

# I am editing this to insert my own Dataset into the dataloader
def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):
    h5_path = osp.join("..", "atlas_data.h5")

    split_df = pd.read_csv(osp.join("..", "atlas_cross_val_index.csv"))
    val_mask = split_df["cross_val"] == 0 # change to whatever cross val sets you want

    random_indices = split_df[~val_mask]["random_indices"].to_list()
    
    val_groups = split_df[val_mask]["pdb"].to_list()
    train_groups = split_df[~val_mask]["pdb"].to_list()

    train_set = H5Dataset(h5_path, random_indices=random_indices, groups=train_groups)
    valid_set = H5Dataset(h5_path, random_indices=None, groups=val_groups)
    test_set = H5Dataset(h5_path, random_indices=None, groups=[])

    assert test_set.__len__() == 0, f"test set should have len 0, has len {test_set.__len__()}"
    
    collate_fn = featurize_GTrans

    train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader_GTrans(valid_set, batch_size=200, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader_GTrans(test_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn)

    return test_loader
