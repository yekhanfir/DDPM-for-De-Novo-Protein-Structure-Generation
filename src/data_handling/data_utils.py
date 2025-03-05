from src import (
   ATOM_TYPES,
   RESTYPES,
   RESTYPE_1TO3, 
   RESTYPE_NUM, 
   PDB_MAX_CHAINS, 
   PDB_CHAIN_IDS, 
   Protein
)
import numpy as np
import pandas as pd
import torch


def make_np_example(coords_dict):
    """Make a dictionary of non-batched numpy protein features."""
    bb_ATOM_TYPES = ['N', 'CA', 'C', 'O']
    bb_idx = [i for i, atom_type in enumerate(ATOM_TYPES)
              if atom_type in bb_ATOM_TYPES]

    num_res = np.array(coords_dict['N']).shape[0]
    atom_positions = np.zeros([num_res, 37, 3], dtype=float)

    for i, atom_type in enumerate(ATOM_TYPES):
        if atom_type in bb_ATOM_TYPES:
            atom_positions[:, i, :] = np.array(coords_dict[atom_type])

    # Mask nan / None coordinates.
    nan_pos = np.isnan(atom_positions)[..., 0]
    atom_positions[nan_pos] = 0.
    atom_mask = np.zeros([num_res, 37])
    atom_mask[..., bb_idx] = 1
    atom_mask[nan_pos] = 0

    batch = {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask,
        'residue_index': np.arange(num_res)
    }
    return batch


def make_fixed_size(np_example, max_seq_length=500):
    """Pad features to fixed sequence length, i.e. currently axis=0."""
    for k, v in np_example.items():
        pad = max_seq_length - v.shape[0]
        if pad > 0:
            v = np.pad(v, ((0, pad),) + ((0, 0),) * (len(v.shape) - 1))
        elif pad < 0:
            v = v[:max_seq_length]
        np_example[k] = v


def center_positions(np_example):
  """Center 'atom_positions' on CA center of mass."""
  atom_positions = np_example['atom_positions']
  atom_mask = np_example['atom_mask']
  ca_positions = atom_positions[:, 1, :]
  ca_mask = atom_mask[:, 1]

  ca_center = (np.sum(ca_mask[..., None] * ca_positions, axis=0) /
   (np.sum(ca_mask, axis=0) + 1e-9))
  atom_positions = ((atom_positions - ca_center[None, ...]) *
                    atom_mask[..., None])
  np_example['atom_positions'] = atom_positions


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


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


def to_pdb(prot: Protein) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  res_1to3 = lambda r: RESTYPE_1TO3.get(RESTYPES[r], 'UNK')

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > RESTYPE_NUM):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(
          atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
          residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(
        ATOM_TYPES, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def read_data_from_json(data_file_path, splits_file_path):
    print('Reading data...')
    df = pd.read_json(data_file_path, lines=True)
    cath_splits = pd.read_json(splits_file_path, lines=True)
    print('Done.')
    return df, cath_splits