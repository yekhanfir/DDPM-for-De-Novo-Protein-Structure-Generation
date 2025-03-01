import torch
import numpy as np
from src import ATOM_TYPES, Protein


def get_np_structure(sample):
        """
        Return the np_structure to build an instance of the Protein class.
        This was already implemented in the initial notebook, but decided
        to make it part of the DDPM class for convenience.
        """
        # to make this function compatibile with tensor
        # and numpy inputs for general use
        if isinstance(sample, torch.Tensor):
          sample = sample.detach().cpu().numpy()
        bb_atom_types = ['N', 'CA', 'C', 'O']
        bb_idx = [i for i, atom_type in enumerate(ATOM_TYPES)
                  if atom_type in bb_atom_types]

        num_res = len(sample)
        nan_pos = np.isnan(sample)[..., 0]
        sample[nan_pos] = 0.
        atom_mask = np.zeros([num_res, 37])
        atom_mask[..., bb_idx] = 1

        np_sample = {
            'atom_positions': sample,
            'residue_index':np.arange(num_res),
            'atom_mask': atom_mask,
        }
        return np_sample

def get_protein(x, max_seq_length):
        """
        Returns a Protein instance from a given input.
        Expects Tensor or Numpy ndarray.
        first generates the np_structure, then returns a Protein instance.
        """
        np_sample = get_np_structure(x)
        prot = Protein(
            atom_positions=np_sample['atom_positions'],
            atom_mask=np_sample['atom_mask'],
            residue_index=np_sample['residue_index'],
            aatype=np.zeros([max_seq_length,], dtype=np.int32),
            chain_index=np.zeros([max_seq_length,], dtype=np.int32),
            b_factors=np.ones([max_seq_length, 37], dtype=np.int32)
        )
        return prot