import pickle
from data_handling.data_utils import (
    get_protein,
    to_pdb, 
)
from src import *

# load the example proteins we saved during training 
with open('example_proteins.pkl', 'rb') as f:
    example_proteins = pickle.load(f)

sample = example_proteins['original'][0].detach().cpu().numpy()
prot = get_protein(sample)
pdb_str = to_pdb(prot)

with open('sample.pdb', 'w') as f:
   f.write(pdb_str)