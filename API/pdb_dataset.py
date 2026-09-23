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
import re

BACKBONE_ATOMS = ["CA", "N", "C", "O"]
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def download_pdb(pdb_id: str, timeout: int = 10) -> str:
    max_tries = 3
    backoff = 1
    for _ in range(max_tries):
        try:
            if not isinstance(pdb_id, str) or not re.fullmatch(r"[0-9A-Za-z]{4}", pdb_id):
                raise ValueError(f"Invalid PDB ID '{pdb_id}'. Must be 4 alphanumeric characters.")

            url = RCSB_URL.format(pdb_id=pdb_id.upper())

            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise urllib.error.HTTPError(url, response.status, "HTTP error", response.headers, None)
                    data = response.read()
            except urllib.error.HTTPError as e:
                raise urllib.error.HTTPError(e.url, e.code, f"Failed to download PDB file: {e.reason}", e.headers, e.fp)
            except urllib.error.URLError as e:
                raise urllib.error.URLError(f"Network error while downloading PDB file: {e.reason}")

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Downloaded file is not valid UTF-8 text.")

            if not text.strip():
                raise ValueError(f"PDB file for ID '{pdb_id}' is empty.")

            return text
        except ValueError:
            sleep(backoff)
            backoff *= 2
    raise ValueError("reached max backoff")

def extract_backbone(pdb_text: str, pdb_chain_id: str) -> torch.Tensor:
    try:
        pdb_id, chain_id = pdb_chain_id.split("_", 1)
    except ValueError:
        raise ValueError(f"Invalid format '{pdb_chain_id}'. Expected 'PPPP_C'.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())

    if chain_id not in model:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32)

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
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32)

    return torch.tensor(coords, dtype=torch.float32), seq

class PDBDataset(data.Dataset):
    def __init__(self, pdbs, frmat="PPPP_C"):
        self.pdbs = pdbs
        if frmat != "PPPP_C":
            self.pdbs = [pdb[:4] + "_" + pdb[4:5] for pdb in self.pdbs]

        self.pdbs = [(pdb, extract_backbone(download_pdb(pdb[:4]), pdb)) for pdb in tqdm(self.pdbs)]

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_id, (bb_tensor, y) = self.pdbs[idx]
        if any(y_char not in alphabet for y_char in y):
            print(y, alphabet)
        
        return {'title':(pdb_id), 'seq':y, "scores":0} | {atom: bb_tensor[:,i] for i, atom in enumerate(BACKBONE_ATOMS)}
