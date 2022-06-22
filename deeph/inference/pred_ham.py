import json
import os
import time
import warnings
from typing import Union, List
import sys

import tqdm
from configparser import ConfigParser
import numpy as np
from pymatgen.core.structure import Structure
import torch
import torch.autograd.forward_ad as fwAD
import h5py

from deeph import get_graph, DeepHKernal, collate_fn, write_ham_h5, load_orbital_types, Rotate, dtype_dict, get_rc


def predict(input_dir: str, output_dir: str, disable_cuda: bool, device: str,
            huge_structure: bool, restore_blocks_py: bool, trained_model_dirs: Union[str, List[str]]):
    atom_num_orbital = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat'))
    if isinstance(trained_model_dirs, str):
        trained_model_dirs = [trained_model_dirs]
    assert isinstance(trained_model_dirs, list)
    os.makedirs(output_dir, exist_ok=True)
    predict_spinful = None

    with torch.no_grad():
        read_structure_flag = False
        if restore_blocks_py:
            hoppings_pred = {}
        else:
            index_model = 0
            block_without_restoration = {}
            os.makedirs(os.path.join(output_dir, 'block_without_restoration'), exist_ok=True)
        for trained_model_dir in tqdm.tqdm(trained_model_dirs):
            old_version = False
            assert os.path.exists(os.path.join(trained_model_dir, 'config.ini'))
            if os.path.exists(os.path.join(trained_model_dir, 'best_model.pt')) is False:
                old_version = True
                assert os.path.exists(os.path.join(trained_model_dir, 'best_model.pkl'))
                assert os.path.exists(os.path.join(trained_model_dir, 'src'))

            config = ConfigParser()
            config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
            config.read(os.path.join(trained_model_dir, 'config.ini'))
            config.set('basic', 'save_dir', os.path.join(output_dir, 'pred_ham_std'))
            config.set('basic', 'disable_cuda', str(disable_cuda))
            config.set('basic', 'device', str(device))
            config.set('basic', 'save_to_time_folder', 'False')
            config.set('basic', 'tb_writer', 'False')
            config.set('train', 'pretrained', '')
            config.set('train', 'resume', '')

            kernal = DeepHKernal(config)
            if old_version is False:
                checkpoint = kernal.build_model(trained_model_dir, old_version)
            else:
                warnings.warn('You are using the trained model with an old version')
                checkpoint = torch.load(
                    os.path.join(trained_model_dir, 'best_model.pkl'),
                    map_location=kernal.device
                )
                for key in ['index_to_Z', 'Z_to_index', 'spinful']:
                    if key in checkpoint:
                        setattr(kernal, key, checkpoint[key])
                if hasattr(kernal, 'index_to_Z') is False:
                    kernal.index_to_Z = torch.arange(config.getint('basic', 'max_element') + 1)
                if hasattr(kernal, 'Z_to_index') is False:
                    kernal.Z_to_index = torch.arange(config.getint('basic', 'max_element') + 1)
                if hasattr(kernal, 'spinful') is False:
                    kernal.spinful = False
                kernal.num_species = len(kernal.index_to_Z)
                print("=> load best checkpoint (epoch {})".format(checkpoint['epoch']))
                print(f"=> Atomic types: {kernal.index_to_Z.tolist()}, "
                      f"spinful: {kernal.spinful}, the number of atomic types: {len(kernal.index_to_Z)}.")
                kernal.build_model(trained_model_dir, old_version)
                kernal.model.load_state_dict(checkpoint['state_dict'])

            if predict_spinful is None:
                predict_spinful = kernal.spinful
            else:
                assert predict_spinful == kernal.spinful, "Different models' spinful are not compatible"

            if read_structure_flag is False:
                read_structure_flag = True
                structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T,
                                      np.loadtxt(os.path.join(input_dir, 'element.dat')),
                                      np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T,
                                      coords_are_cartesian=True,
                                      to_unit_cell=False)
                cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype())
                frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
                numbers = kernal.Z_to_index[torch.tensor(structure.atomic_numbers)]
                structure.lattice.matrix.setflags(write=True)
                lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
                inv_lattice = torch.inverse(lattice)

                if os.path.exists(os.path.join(input_dir, 'graph.pkl')):
                    data = torch.load(os.path.join(input_dir, 'graph.pkl'))
                    print(f"Load processed graph from {os.path.join(input_dir, 'graph.pkl')}")
                else:
                    begin = time.time()
                    data = get_graph(cart_coords, frac_coords, numbers, 0,
                                     r=kernal.config.getfloat('graph', 'radius'),
                                     max_num_nbr=kernal.config.getint('graph', 'max_num_nbr'),
                                     numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                                     tb_folder=input_dir, interface="h5_rc_only",
                                     num_l=kernal.config.getint('network', 'num_l'),
                                     create_from_DFT=kernal.config.getboolean('graph', 'create_from_DFT',
                                                                              fallback=True),
                                     if_lcmp_graph=kernal.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                                     separate_onsite=kernal.separate_onsite,
                                     target=kernal.config.get('basic', 'target'), huge_structure=huge_structure,
                                     if_new_sp=kernal.config.getboolean('graph', 'new_sp', fallback=False),
                                     )
                    torch.save(data, os.path.join(input_dir, 'graph.pkl'))
                    print(
                        f"Save processed graph to {os.path.join(input_dir, 'graph.pkl')}, cost {time.time() - begin} seconds")
                batch, subgraph = collate_fn([data])
                sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

            output = kernal.model(batch.x.to(kernal.device), batch.edge_index.to(kernal.device),
                                  batch.edge_attr.to(kernal.device),
                                  batch.batch.to(kernal.device),
                                  sub_atom_idx.to(kernal.device), sub_edge_idx.to(kernal.device),
                                  sub_edge_ang.to(kernal.device), sub_index.to(kernal.device),
                                  huge_structure=huge_structure)
            output = output.detach().cpu()
            if restore_blocks_py:
                for index in range(batch.edge_attr.shape[0]):
                    R = torch.round(batch.edge_attr[index, 4:7] @ inv_lattice - batch.edge_attr[index, 7:10] @ inv_lattice).int().tolist()
                    i, j = batch.edge_index[:, index]
                    key_term = (*R, i.item() + 1, j.item() + 1)
                    key_term = str(list(key_term))
                    for index_orbital, orbital_dict in enumerate(kernal.orbital):
                        if f'{kernal.index_to_Z[numbers[i]].item()} {kernal.index_to_Z[numbers[j]].item()}' not in orbital_dict:
                            continue
                        orbital_i, orbital_j = orbital_dict[f'{kernal.index_to_Z[numbers[i]].item()} {kernal.index_to_Z[numbers[j]].item()}']

                        if not key_term in hoppings_pred:
                            if kernal.spinful:
                                hoppings_pred[key_term] = np.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), np.nan + np.nan * (1j))
                            else:
                                hoppings_pred[key_term] = np.full((atom_num_orbital[i], atom_num_orbital[j]), np.nan)
                        if kernal.spinful:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital * 8 + 0] + output[index][index_orbital * 8 + 1] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 2] + output[index][index_orbital * 8 + 3] * 1j
                            hoppings_pred[key_term][orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 4] + output[index][index_orbital * 8 + 5] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, orbital_j] = output[index][index_orbital * 8 + 6] + output[index][index_orbital * 8 + 7] * 1j
                        else:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital]  # about output shape w/ or w/o soc, see graph.py line 164, and kernel.py line 281.
            else:
                if 'edge_index' not in block_without_restoration:
                    assert index_model == 0
                    block_without_restoration['edge_index'] = batch.edge_index
                    block_without_restoration['edge_attr'] = batch.edge_attr
                block_without_restoration[f'output_{index_model}'] = output.numpy()
                with open(os.path.join(output_dir, 'block_without_restoration', f'orbital_{index_model}.json'), 'w') as orbital_f:
                    json.dump(kernal.orbital, orbital_f, indent=4)
                index_model += 1
            sys.stdout = sys.stdout.terminal
            sys.stderr = sys.stderr.terminal

        if restore_blocks_py:
            for hamiltonian in hoppings_pred.values():
                assert np.all(np.isnan(hamiltonian) == False)
            write_ham_h5(hoppings_pred, path=os.path.join(output_dir, 'rh_pred.h5'))
        else:
            block_without_restoration['num_model'] = index_model
            write_ham_h5(block_without_restoration, path=os.path.join(output_dir, 'block_without_restoration', 'block_without_restoration.h5'))
        with open(os.path.join(output_dir, "info.json"), 'w') as info_f:
            json.dump({
                "isspinful": predict_spinful
            }, info_f)


def predict_with_grad(input_dir: str, output_dir: str, disable_cuda: bool, device: str,
                      huge_structure: bool, trained_model_dirs: Union[str, List[str]]):
    atom_num_orbital, orbital_types = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat'), return_orbital_types=True)

    if isinstance(trained_model_dirs, str):
        trained_model_dirs = [trained_model_dirs]
    assert isinstance(trained_model_dirs, list)
    os.makedirs(output_dir, exist_ok=True)
    predict_spinful = None

    read_structure_flag = False
    rh_dict = {}
    hamiltonians_pred = {}
    hamiltonians_grad_pred = {}

    for trained_model_dir in tqdm.tqdm(trained_model_dirs):
        old_version = False
        assert os.path.exists(os.path.join(trained_model_dir, 'config.ini'))
        if os.path.exists(os.path.join(trained_model_dir, 'best_model.pt')) is False:
            old_version = True
            assert os.path.exists(os.path.join(trained_model_dir, 'best_model.pkl'))
            assert os.path.exists(os.path.join(trained_model_dir, 'src'))

        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
        config.read(os.path.join(trained_model_dir, 'config.ini'))
        config.set('basic', 'save_dir', os.path.join(output_dir, 'pred_ham_std'))
        config.set('basic', 'disable_cuda', str(disable_cuda))
        config.set('basic', 'device', str(device))
        config.set('basic', 'save_to_time_folder', 'False')
        config.set('basic', 'tb_writer', 'False')
        config.set('train', 'pretrained', '')
        config.set('train', 'resume', '')

        kernal = DeepHKernal(config)
        if old_version is False:
            checkpoint = kernal.build_model(trained_model_dir, old_version)
        else:
            warnings.warn('You are using the trained model with an old version')
            checkpoint = torch.load(
                os.path.join(trained_model_dir, 'best_model.pkl'),
                map_location=kernal.device
            )
            for key in ['index_to_Z', 'Z_to_index', 'spinful']:
                if key in checkpoint:
                    setattr(kernal, key, checkpoint[key])
            if hasattr(kernal, 'index_to_Z') is False:
                kernal.index_to_Z = torch.arange(config.getint('basic', 'max_element') + 1)
            if hasattr(kernal, 'Z_to_index') is False:
                kernal.Z_to_index = torch.arange(config.getint('basic', 'max_element') + 1)
            if hasattr(kernal, 'spinful') is False:
                kernal.spinful = False
            kernal.num_species = len(kernal.index_to_Z)
            print("=> load best checkpoint (epoch {})".format(checkpoint['epoch']))
            print(f"=> Atomic types: {kernal.index_to_Z.tolist()}, "
                  f"spinful: {kernal.spinful}, the number of atomic types: {len(kernal.index_to_Z)}.")
            kernal.build_model(trained_model_dir, old_version)
            kernal.model.load_state_dict(checkpoint['state_dict'])

        if predict_spinful is None:
            predict_spinful = kernal.spinful
        else:
            assert predict_spinful == kernal.spinful, "Different models' spinful are not compatible"

        if read_structure_flag is False:
            read_structure_flag = True
            structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T,
                                  np.loadtxt(os.path.join(input_dir, 'element.dat')),
                                  np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T,
                                  coords_are_cartesian=True,
                                  to_unit_cell=False)
            cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype(), requires_grad=True, device=kernal.device)
            num_atom = cart_coords.shape[0]
            frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
            numbers = kernal.Z_to_index[torch.tensor(structure.atomic_numbers)]
            structure.lattice.matrix.setflags(write=True)
            lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
            inv_lattice = torch.inverse(lattice)

            fid_rc = get_rc(input_dir, None, radius=-1, create_from_DFT=True, if_require_grad=True, cart_coords=cart_coords)

            assert kernal.config.getboolean('graph', 'new_sp', fallback=False)
            data = get_graph(cart_coords.to(kernal.device), frac_coords, numbers, 0,
                             r=kernal.config.getfloat('graph', 'radius'),
                             max_num_nbr=kernal.config.getint('graph', 'max_num_nbr'),
                             numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                             tb_folder=input_dir, interface="h5_rc_only",
                             num_l=kernal.config.getint('network', 'num_l'),
                             create_from_DFT=kernal.config.getboolean('graph', 'create_from_DFT', fallback=True),
                             if_lcmp_graph=kernal.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                             separate_onsite=kernal.separate_onsite,
                             target=kernal.config.get('basic', 'target'), huge_structure=huge_structure,
                             if_new_sp=True, if_require_grad=True, fid_rc=fid_rc)
            batch, subgraph = collate_fn([data])
            sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

            torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[torch.get_default_dtype()]
            rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real,
                                   torch_dtype_complex=torch_dtype_complex,
                                   device=kernal.device, spinful=kernal.spinful)

        output = kernal.model(batch.x, batch.edge_index.to(kernal.device),
                              batch.edge_attr,
                              batch.batch.to(kernal.device),
                              sub_atom_idx.to(kernal.device), sub_edge_idx.to(kernal.device),
                              sub_edge_ang, sub_index.to(kernal.device),
                              huge_structure=huge_structure)

        index_for_matrix_block_real_dict = {}  # key is atomic number pair
        if kernal.spinful:
            index_for_matrix_block_imag_dict = {}  # key is atomic number pair

        for index in range(batch.edge_attr.shape[0]):
            R = torch.round(batch.edge_attr[index, 4:7].cpu() @ inv_lattice - batch.edge_attr[index, 7:10].cpu() @ inv_lattice).int().tolist()
            i, j = batch.edge_index[:, index]
            key_tensor = torch.tensor([*R, i, j])
            numbers_pair = (kernal.index_to_Z[numbers[i]].item(), kernal.index_to_Z[numbers[j]].item())
            if numbers_pair not in index_for_matrix_block_real_dict:
                if not kernal.spinful:
                    index_for_matrix_block_real = torch.full((atom_num_orbital[i], atom_num_orbital[j]), -1)
                else:
                    index_for_matrix_block_real = torch.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), -1)
                    index_for_matrix_block_imag = torch.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), -1)
                for index_orbital, orbital_dict in enumerate(kernal.orbital):
                    if f'{kernal.index_to_Z[numbers[i]].item()} {kernal.index_to_Z[numbers[j]].item()}' not in orbital_dict:
                        continue
                    orbital_i, orbital_j = orbital_dict[f'{kernal.index_to_Z[numbers[i]].item()} {kernal.index_to_Z[numbers[j]].item()}']
                    if not kernal.spinful:
                        index_for_matrix_block_real[orbital_i, orbital_j] = index_orbital
                    else:
                        index_for_matrix_block_real[orbital_i, orbital_j] = index_orbital * 8 + 0
                        index_for_matrix_block_imag[orbital_i, orbital_j] = index_orbital * 8 + 1
                        index_for_matrix_block_real[atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 2
                        index_for_matrix_block_imag[atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 3
                        index_for_matrix_block_real[orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 4
                        index_for_matrix_block_imag[orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 5
                        index_for_matrix_block_real[atom_num_orbital[i] + orbital_i, orbital_j] = index_orbital * 8 + 6
                        index_for_matrix_block_imag[atom_num_orbital[i] + orbital_i, orbital_j] = index_orbital * 8 + 7
                assert torch.all(index_for_matrix_block_real != -1), 'json string "orbital" should be complete for Hamiltonian grad'
                if kernal.spinful:
                    assert torch.all(index_for_matrix_block_imag != -1), 'json string "orbital" should be complete for Hamiltonian grad'

                index_for_matrix_block_real_dict[numbers_pair] = index_for_matrix_block_real
                if kernal.spinful:
                    index_for_matrix_block_imag_dict[numbers_pair] = index_for_matrix_block_imag
            else:
                index_for_matrix_block_real = index_for_matrix_block_real_dict[numbers_pair]
                if kernal.spinful:
                    index_for_matrix_block_imag = index_for_matrix_block_imag_dict[numbers_pair]

            if not kernal.spinful:
                rh_dict[key_tensor] = output[index][index_for_matrix_block_real]
            else:
                rh_dict[key_tensor] = output[index][index_for_matrix_block_real] + 1j * output[index][index_for_matrix_block_imag]

        sys.stdout = sys.stdout.terminal
        sys.stderr = sys.stderr.terminal

    print("=> Hamiltonian has been predicted, calculate the grad...")
    for key_tensor, rotated_hamiltonian in tqdm.tqdm(rh_dict.items()):
        atom_i = key_tensor[3]
        atom_j = key_tensor[4]
        assert atom_i >= 0
        assert atom_i < num_atom
        assert atom_j >= 0
        assert atom_j < num_atom
        key_str = str(list([key_tensor[0].item(), key_tensor[1].item(), key_tensor[2].item(), atom_i.item() + 1, atom_j.item() + 1]))
        assert key_str in fid_rc, f'Can not found the key "{key_str}" in rc.h5'
        # rotation_matrix = torch.tensor(fid_rc[key_str], dtype=torch_dtype_real, device=kernal.device).T
        rotation_matrix = fid_rc[key_str].T
        hamiltonian = rotate_kernel.rotate_openmx_H(rotated_hamiltonian, rotation_matrix, orbital_types[atom_i], orbital_types[atom_j])
        hamiltonians_pred[key_str] = hamiltonian.detach().cpu()
        assert kernal.spinful is False  # 检查soc时是否正确
        assert len(hamiltonian.shape) == 2
        dim_1, dim_2 = hamiltonian.shape[:]
        assert key_str not in hamiltonians_grad_pred
        if not kernal.spinful:
            hamiltonians_grad_pred[key_str] = np.full((dim_1, dim_2, num_atom, 3), np.nan)
        else:
            hamiltonians_grad_pred[key_str] = np.full((2 * dim_1, 2 * dim_2, num_atom, 3), np.nan + 1j * np.nan)

    write_ham_h5(hamiltonians_pred, path=os.path.join(output_dir, 'hamiltonians_pred.h5'))
    write_ham_h5(hamiltonians_grad_pred, path=os.path.join(output_dir, 'hamiltonians_grad_pred.h5'))
    with open(os.path.join(output_dir, "info.json"), 'w') as info_f:
        json.dump({
            "isspinful": predict_spinful
        }, info_f)
    fid_rc.close()
