"""
Author: Jens Lundsgaard
Based on: cath_dataset.py
"""
import os
import json
import traceback
import numpy as np
import torch.utils.data as data
import h5py
import torch
import requests
from Bio.SeqUtils import seq1
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from tqdm import tqdm
import pandas as pd
import io
from time import sleep

import urllib.request
import urllib.error
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

BACKBONE_ATOMS = ["CA", "N", "C", "O"]
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
alphabet = 'ACDEFGHIKLMNPQRSTVWY'
CACHE_DIR = "cif_cache"

def download_pdb_cached(pdb_id: str, timeout: int = 10) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{pdb_id.upper()}.cif")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    text = download_pdb(pdb_id, timeout=timeout)
    with open(path, "w") as f:
        f.write(text)
    return text


class PDBDataset(data.Dataset):
    def __init__(self, pdbs, frmat="PPPP_C", max_workers=16):
        self._pdbs = pdbs
        if frmat != "PPPP_C":
            self._pdbs = [pdb[:4] + "_" + pdb[4:5] for pdb in self._pdbs]

        unique_ids = sorted(set(pdb[:4] for pdb in self._pdbs))

        cif_text = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(download_pdb_cached, pid): pid for pid in unique_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
                pid = futures[fut]
                try:
                    cif_text[pid] = fut.result()
                except Exception as e:
                    print(f"Failed to download {pid}: {e}")

        self.pdbs = []
        for pdb in tqdm(self._pdbs, desc="parsing"):
            pid = pdb[:4]
            if pid not in cif_text:
                continue  # download failed, already logged
            try:
                result = extract_backbone(cif_text[pid], pdb)
                self.pdbs.append((pdb, result))
            except ValueError as e:
                print(f"Skipping {pdb}: {e}")
    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_id, (bb_tensor, y) = self.pdbs[idx]
        print(y, alphabet)
        
        return {'title':(pdb_id, -1), 'seq':y} | {atom: bb_tensor[:,i] for i, atom in enumerate(BACKBONE_ATOMS)}
