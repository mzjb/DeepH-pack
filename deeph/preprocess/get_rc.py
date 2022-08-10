import os
import json

import h5py
import numpy as np
import torch


class Neighbours:
    def __init__(self):
        self.Rs = []
        self.dists = []
        self.eijs = []
        self.indices = []

    def __str__(self):
        return 'Rs: {}\ndists: {}\neijs: {}\nindices: {}'.format(
            self.Rs, self.dists, self.indices, self.eijs)


def _get_local_coordinate(eij, neighbours_i, gen_rc_idx=False, atom_j=None, atom_j_R=None, r2_rand=False):
    if gen_rc_idx:
        rc_idx = np.full(8, np.nan, dtype=np.int32)
        assert r2_rand is False
        assert atom_j is not None, 'atom_j must be specified when gen_rc_idx is True'
        assert atom_j_R is not None, 'atom_j_R must be specified when gen_rc_idx is True'
    else:
        rc_idx = None
    if r2_rand:
        r2_list = []

    if not np.allclose(eij.detach(), torch.zeros_like(eij)):
        r1 = eij
        if gen_rc_idx:
            rc_idx[0] = atom_j
            rc_idx[1:4] = atom_j_R
    else:
        r1 = neighbours_i.eijs[1]
        if gen_rc_idx:
            rc_idx[0] = neighbours_i.indices[1]
            rc_idx[1:4] = neighbours_i.Rs[1]
    r2_flag = None
    for r2, r2_index, r2_R in zip(neighbours_i.eijs[1:], neighbours_i.indices[1:], neighbours_i.Rs[1:]):
        if torch.norm(torch.cross(r1, r2)) > 1e-6:
            if gen_rc_idx:
                rc_idx[4] = r2_index
                rc_idx[5:8] = r2_R
            r2_flag = True
            if r2_rand:
                if (len(r2_list) == 0) or (torch.norm(r2_list[0]) + 0.5 > torch.norm(r2)):
                    r2_list.append(r2)
                else:
                    break
            else:
                break
    assert r2_flag is not None, "There is no linear independent chemical bond in the Rcut range, this may be caused by a too small Rcut or the structure is 1D"
    if r2_rand:
        # print(f"r2 is randomly chosen from {len(r2_list)} candidates")
        r2 = r2_list[np.random.randint(len(r2_list))]
    local_coordinate_1 = r1 / torch.norm(r1)
    local_coordinate_2 = torch.cross(r1, r2) / torch.norm(torch.cross(r1, r2))
    local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2)
    return torch.stack([local_coordinate_1, local_coordinate_2, local_coordinate_3], dim=-1), rc_idx


def get_rc(input_dir, output_dir, radius, r2_rand=False, gen_rc_idx=False, gen_rc_by_idx="", create_from_DFT=True, neighbour_file='overlaps.h5', if_require_grad=False, cart_coords=None):
    if not if_require_grad:
        assert os.path.exists(os.path.join(input_dir, 'site_positions.dat')), 'No site_positions.dat found in {}'.format(input_dir)
        cart_coords = torch.tensor(np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T)
    else:
        assert cart_coords is not None, 'cart_coords must be provided if "if_require_grad" is True'
    assert os.path.exists(os.path.join(input_dir, 'lat.dat')), 'No lat.dat found in {}'.format(input_dir)
    lattice = torch.tensor(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T, dtype=cart_coords.dtype)

    rc_dict = {}
    if gen_rc_idx:
        assert r2_rand is False, 'r2_rand must be False when gen_rc_idx is True'
        assert gen_rc_by_idx == "", 'gen_rc_by_idx must be "" when gen_rc_idx is True'
        rc_idx_dict = {}
    neighbours_dict = {}
    if gen_rc_by_idx != "":
        # print(f'get local coordinate using {os.path.join(gen_rc_by_idx, "rc_idx.h5")} from: {input_dir}')
        assert os.path.exists(os.path.join(gen_rc_by_idx, "rc_idx.h5")), 'Atomic indices for constructing rc rc_idx.h5 is not found in {}'.format(gen_rc_by_idx)
        fid_rc_idx = h5py.File(os.path.join(gen_rc_by_idx, "rc_idx.h5"), 'r')
        for key_str, rc_idx in fid_rc_idx.items():
            key = json.loads(key_str)
            # R = torch.tensor([key[0], key[1], key[2]])
            atom_i = key[3] - 1
            cart_coords_i = cart_coords[atom_i]

            r1 = cart_coords[rc_idx[0]] + torch.tensor(rc_idx[1:4]).type(cart_coords.dtype) @ lattice - cart_coords_i
            r2 = cart_coords[rc_idx[4]] + torch.tensor(rc_idx[5:8]).type(cart_coords.dtype) @ lattice - cart_coords_i
            local_coordinate_1 = r1 / torch.norm(r1)
            local_coordinate_2 = torch.cross(r1, r2) / torch.norm(torch.cross(r1, r2))
            local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2)

            rc_dict[key_str] = torch.stack([local_coordinate_1, local_coordinate_2, local_coordinate_3], dim=-1)
        fid_rc_idx.close()
    else:
        # print("get local coordinate from:", input_dir)
        if create_from_DFT:
            assert os.path.exists(os.path.join(input_dir, neighbour_file)), 'No {} found in {}'.format(neighbour_file, input_dir)
            fid_OLP = h5py.File(os.path.join(input_dir, neighbour_file), 'r')
            for key_str in fid_OLP.keys():
                key = json.loads(key_str)
                R = torch.tensor([key[0], key[1], key[2]])
                atom_i = key[3] - 1
                atom_j = key[4] - 1
                cart_coords_i = cart_coords[atom_i]
                cart_coords_j = cart_coords[atom_j] + R.type(cart_coords.dtype) @ lattice
                eij = cart_coords_j - cart_coords_i
                dist = torch.norm(eij)
                if radius > 0 and dist > radius:
                    continue
                if atom_i not in neighbours_dict:
                    neighbours_dict[atom_i] = Neighbours()
                neighbours_dict[atom_i].Rs.append(R)
                neighbours_dict[atom_i].dists.append(dist)
                neighbours_dict[atom_i].eijs.append(eij)
                neighbours_dict[atom_i].indices.append(atom_j)

            for atom_i, neighbours_i in neighbours_dict.items():
                neighbours_i.Rs = torch.stack(neighbours_i.Rs)
                neighbours_i.dists = torch.tensor(neighbours_i.dists, dtype=cart_coords.dtype)
                neighbours_i.eijs = torch.stack(neighbours_i.eijs)
                neighbours_i.indices = torch.tensor(neighbours_i.indices)

                neighbours_i.dists, sorted_index = torch.sort(neighbours_i.dists)
                neighbours_i.Rs = neighbours_i.Rs[sorted_index]
                neighbours_i.eijs = neighbours_i.eijs[sorted_index]
                neighbours_i.indices = neighbours_i.indices[sorted_index]

                assert np.allclose(neighbours_i.eijs[0].detach(), torch.zeros_like(neighbours_i.eijs[0])), 'eijs[0] should be zero'

                for R, eij, atom_j, atom_j_R in zip(neighbours_i.Rs, neighbours_i.eijs, neighbours_i.indices, neighbours_i.Rs):
                    key_str = str(list([*R.tolist(), atom_i + 1, atom_j.item() + 1]))
                    if gen_rc_idx:
                        rc_dict[key_str], rc_idx_dict[key_str] = _get_local_coordinate(eij, neighbours_i, gen_rc_idx, atom_j, atom_j_R)
                    else:
                        rc_dict[key_str] = _get_local_coordinate(eij, neighbours_i, r2_rand=r2_rand)[0]
        else:
            raise NotImplementedError

        if create_from_DFT:
            fid_OLP.close()

    if if_require_grad:
        return rc_dict
    else:
        if os.path.exists(os.path.join(output_dir, 'rc_julia.h5')):
            rc_old_flag = True
            fid_rc_old = h5py.File(os.path.join(output_dir, 'rc_julia.h5'), 'r')
        else:
            rc_old_flag = False
        fid_rc = h5py.File(os.path.join(output_dir, 'rc.h5'), 'w')
        for k, v in rc_dict.items():
            if rc_old_flag:
                assert np.allclose(v, fid_rc_old[k][...], atol=1e-4), f"{k}, {v}, {fid_rc_old[k][...]}"
            fid_rc[k] = v
        fid_rc.close()
        if gen_rc_idx:
            fid_rc_idx = h5py.File(os.path.join(output_dir, 'rc_idx.h5'), 'w')
            for k, v in rc_idx_dict.items():
                fid_rc_idx[k] = v
            fid_rc_idx.close()
