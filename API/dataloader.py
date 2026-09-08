import copy
import os.path as osp

from API.cath_dataset import CATH
from API.ts_dataset import TS

from API.dataloader_gtrans import DataLoader_GTrans
from API.featurizer import featurize_GTrans

# I am editing this to insert my own Dataset into the dataloader
def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn)

    return test_loader
