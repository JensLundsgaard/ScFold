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
import requests
from Bio.SeqUtils import seq1
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from tqdm import tqdm
import pandas as pd
from h5_dataset import BACKBONE_ATOMS 
BACKBONE_ATOMS = ["CA", "N", "C", "O"]

def download_pdb(pdb_id: str) -> str:
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def extract_backbone(pdb_text: str, pdb_id: str, chain:str) -> torch.Tensor:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())
    residues = []
    for chain in model:
        if chain chain):
            for residue in chain:
                if not residue.has_id("CA"):
                    continue
                try:
                    atom_coords = [residue[atom].coord for atom in BACKBONE_ATOMS]
                except KeyError:
                    continue
                coords.append(atom_coords)

    if not coords:
        raise ValueError(f"No complete backbone residues found in {pdb_id}")

    return torch.tensor(coords, dtype=torch.float32), 



class PDBDataset(Dataset):
    def __init__(self, pdbs, frmat="PPPP_C"):
        self.pdbs = pdbs
        if frmat != "PPPP_C":
            self.pdbs = [pdb[:4] + "_" + pdb[4:5] for pdb in self.pdbs]

        self.pdbs = self.pdbs.map(lambda pdb: (pdb, extract_backbone(download_pdb(pdb[:4]), pdb[:4], pdb[5:6])))

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_id, tensor = self.pdbs[idx]
        
        return {'title':(pdb_id), 'seq':y} | {atom: coordinates[:,i] for i, atom in enumerate(self.__class__.BACKBONE_ATOMS)}
