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
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU","LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

import torch
import torch.utils.data as data
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1


def cache_path(pdb_id: str, cache_dir=os.path.join("..","pdbs"), use_cath=False) -> str:
    return os.path.join(cache_dir, f"{pdb_id.upper()}" + ("" if use_cath else ".cif"))

def read_cached_cif(pdb_id: str, cache_dir=os.path.join("..","pdbs"), use_cath=False) -> str:
    path = cache_path(pdb_id, cache_dir, use_cath=use_cath)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached .cif for '{pdb_id}' at {path}. "
            f"Run download_cifs.py first to populate the cache."
        )
    with open(path, "r") as f:
        return f.read()


def extract_backbone(pdb_text: str, pdb_chain_id: str, use_cath=False):
    if(not use_cath):
        pdb_chain_id, chain_id = pdb_chain_id.split("_", 1)
    else:
        chain_id = pdb_chain_id[4:5]

    parser = MMCIFParser(QUIET=True) if not use_cath else PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_chain_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())

    if chain_id not in model:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32), ""

    chain = model[chain_id]
    seq = ""
    for residue in chain:
        if residue.id[0] != " ":
            continue

        if not all(atom_name in residue for atom_name in BACKBONE_ATOMS):
            continue
        seq += seq1(residue.get_resname()).upper()
        atom_coords = [residue[atom_name].coord for atom_name in BACKBONE_ATOMS]
        coords.append(atom_coords)

    if not coords:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32), ""
    if any(seq_char not in alphabet for seq_char in seq):
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32), ""

    return torch.tensor(coords, dtype=torch.float32), seq


class PDBDataset(data.Dataset):
    def __init__(self, pdbs, use_cath=True):

        cache_dir=os.path.join("..","pdbs") if not use_cath else os.path.join("..", "dompdb")
        self._pdbs = pdbs

        unique_ids = sorted({(pdb[:4] if not use_cath else pdb) for pdb in self._pdbs})
        cif_text = {}
        for pid in tqdm(unique_ids, desc="reading cache"):
            try:
                cif_text[pid] = read_cached_cif(pid, cache_dir, use_cath=use_cath)
            except FileNotFoundError as e:
                print(e)

        self.pdbs = []
        for pdb in tqdm(self._pdbs, desc="parsing"):
            pid = pdb[:4] if not use_cath else pdb
            if pid not in cif_text:
                continue  # missing from cache, already logged
            try:
                result = extract_backbone(cif_text[pid], pdb, use_cath=use_cath)
                self.pdbs.append((pdb, result))
            except ValueError as e:
                print(f"Skipping {pdb}: {e}")

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_id, (bb_tensor, y) = self.pdbs[idx]

        return {"title": (pdb_id, -1), "seq": y} | {
            atom: bb_tensor[:, i] for i, atom in enumerate(BACKBONE_ATOMS)
        }
