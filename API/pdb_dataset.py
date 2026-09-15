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

class SelectChain(Select):
    def __init__(self, chain):
        super().__init__()
        self.chain = chain
    def accept_chain(self, chain):
        return chain == self.chain

def retrieve_pdb_file(pdb_id, file_format = "cif", parent_dir="./"):
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.{file_format}"

    # TODO remove in case of anonymization
    headers = {
        "User-Agent": "jlundsgaard@wisc.edu"
    }
    file_path = os.path.abspath(os.path.join(parent_dir, f"{pdb_id}.{file_format}"))

    i = 0
    attempts = 8
    delay = 1
    while(i < attempts):
        try:
            if (response := requests.get(url, headers=headers)).status_code != 200:
                time.sleep(delay)
            else:
                with open(file_path, "w") as file:
                    file.write(response.text)
                return file_path
        except requests.exceptions.ReadTimeout:
            time.sleep(delay)
        i += 1
        delay *= 1.3
    raise ValueError(f"{pdb_id} could not be accessed at {url}: error code {response.status_code}")


def load_pdb(pdb_plus_chain, pdb_dir): # save the pdb to a directory, if pdb_dir == "" it doesn't save it
    # pdb ids are sometimes formatted like this
    pdb_plus_chain = pdb_plus_chain.replace(":", "_")
    if("_" not in pdb_plus_chain):
        print(f"no chain id: {pdb_plus_chain}")
        pdb_id = pdb_plus_chain
        chain_id = ""
    else:
        pdb_id, chain_id = pdb_plus_chain.split("_")
    pdb_id = pdb_id[:4].upper()

    file_path = retrieve_pdb_file(pdb_id, file_format="cif")

    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, file_path)

    io = PDB.PDBIO()
    io.set_structure(structure)
    pdb_path = os.path.join(pdb_dir, f"{pdb_plus_chain}.pdb")
    io.save(pdb_path, select=SelectChain(chain_id))
    return pdb_path




class PDBDataset(Dataset):
    def __init__(self, pdbs):
        self.pdbs = pdbs
        self.pdb_dir = os.path.abspath("pdbs")
        os.makedirs(pdb_dir, exist_ok=True)
        self.pdbs = self.pdbs.map(lambda pdb: (pdb, load_pdb(pdb, self.pdb_dir)))
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_id, file_path = self.pdbs[idx]

        structure = self.parser.get_structure(pdb_id, file_path)

        
