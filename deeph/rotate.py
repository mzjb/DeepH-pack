import json
import os.path
import warnings

import numpy as np
import h5py
import torch
from e3nn.o3 import Irrep, Irreps, matrix_to_angles

from deeph import load_orbital_types

dtype_dict = {
    np.float32: (torch.float32, torch.float32, torch.complex64),
    np.float64: (torch.float64, torch.float64, torch.complex128),
    np.complex64: (torch.complex64, torch.float32, torch.complex64),
    np.complex128: (torch.complex128, torch.float64, torch.complex128),
    torch.float32: (torch.float32, torch.float32, torch.complex64),
    torch.float64: (torch.float64, torch.float64, torch.complex128),
    torch.complex64: (torch.complex64, torch.float32, torch.complex64),
    torch.complex128: (torch.complex128, torch.float64, torch.complex128),
}


class Rotate:
    def __init__(self, torch_dtype, torch_dtype_real=torch.float64, torch_dtype_complex=torch.cdouble,
                 device=torch.device('cpu'), spinful=False):
        self.dtype = torch_dtype
        self.torch_dtype_real = torch_dtype_real
        self.device = device
        self.spinful = spinful
        sqrt_2 = 1.4142135623730951
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch_dtype_complex, device=device),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]],
                            dtype=torch_dtype_complex, device=device),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch_dtype_complex, device=device),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch_dtype_complex, device=device),
        }
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=torch_dtype).to(device=device),
            1: torch.eye(3, dtype=torch_dtype)[[1, 2, 0]].to(device=device),
            2: torch.eye(5, dtype=torch_dtype)[[2, 4, 0, 3, 1]].to(device=device),
            3: torch.eye(7, dtype=torch_dtype)[[6, 4, 2, 0, 1, 3, 5]].to(device=device)
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if self.spinful:
            raise NotImplementedError
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H_old(self, H, R, l_lefts, l_rights, order_xyz=True):
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R

        block_lefts = []
        for l_left in l_lefts:
            block_lefts.append(
                self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_left])
        rotation_left = torch.block_diag(*block_lefts)

        block_rights = []
        for l_right in l_rights:
            block_rights.append(
                self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right])
        rotation_right = torch.block_diag(*block_rights)

        return torch.einsum("cd,ca,db->ab", H, rotation_left, rotation_right)

    def rotate_openmx_H(self, H, R, l_lefts, l_rights, order_xyz=True):
        # spin-1/2 is writed by gongxx
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        irreps_left = Irreps([(1, (l, 1)) for l in l_lefts])
        irreps_right = Irreps([(1, (l, 1)) for l in l_rights])
        U_left = irreps_left.D_from_matrix(R_e3nn)
        U_right = irreps_right.D_from_matrix(R_e3nn)
        openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in l_lefts])
        openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in l_rights])
        if self.spinful:
            U_left = torch.kron(self.D_one_half(R_e3nn), U_left)
            U_right = torch.kron(self.D_one_half(R_e3nn), U_right)
            openmx2wiki_left = torch.block_diag(openmx2wiki_left, openmx2wiki_left)
            openmx2wiki_right = torch.block_diag(openmx2wiki_right, openmx2wiki_right)
        return openmx2wiki_left.T @ U_left.transpose(-1, -2).conj() @ openmx2wiki_left @ H \
               @ openmx2wiki_right.T @ U_right @ openmx2wiki_right

    def rotate_openmx_phiVdphi(self, phiVdphi, R, l_lefts, l_rights, order_xyz=True):
        if self.spinful:
            raise NotImplementedError
        assert phiVdphi.shape[-1] == 3
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        block_lefts = []
        for l_left in l_lefts:
            block_lefts.append(
                self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_left])
        rotation_left = torch.block_diag(*block_lefts)

        block_rights = []
        for l_right in l_rights:
            block_rights.append(
                self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right])
        rotation_right = torch.block_diag(*block_rights)

        rotation_x = self.Us_openmx2wiki[1].T @ Irrep(1, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[1]

        return torch.einsum("def,da,eb,fc->abc", phiVdphi, rotation_left, rotation_right, rotation_x)

    def wiki2openmx_H(self, H, l_left, l_right):
        if self.spinful:
            raise NotImplementedError
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        if self.spinful:
            raise NotImplementedError
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def rotate_matrix_convert(self, R):
        return R.index_select(0, R.new_tensor([1, 2, 0]).int()).index_select(1, R.new_tensor([1, 2, 0]).int())

    def D_one_half(self, R):
        # writed by gongxx
        assert self.spinful
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2  # parity index
        alpha, beta, gamma = matrix_to_angles(R)
        J = torch.tensor([[1, 1], [1j, -1j]], dtype=self.dtype) / 1.4142135623730951  # <1/2 mz|1/2 my>
        Uz1 = self._sp_z_rot(alpha)
        Uy = J @ self._sp_z_rot(beta) @ J.T.conj()
        Uz2 = self._sp_z_rot(gamma)
        return Uz1 @ Uy @ Uz2

    def _sp_z_rot(self, angle):
        # writed by gongxx
        assert self.spinful
        M = torch.zeros([*angle.shape, 2, 2], dtype=self.dtype)
        inds = torch.tensor([0, 1])
        freqs = torch.tensor([0.5, -0.5], dtype=self.dtype)
        M[..., inds, inds] = torch.exp(- freqs * (1j) * angle[..., None])
        return M


def get_rh(input_dir, output_dir, target='hamiltonian'):
    torch_device = torch.device('cpu')
    assert target in ['hamiltonian', 'phiVdphi']
    file_name = {
        'hamiltonian': 'hamiltonians.h5',
        'phiVdphi': 'phiVdphi.h5',
    }[target]
    prime_file_name = {
        'hamiltonian': 'rh.h5',
        'phiVdphi': 'rphiVdphi.h5',
    }[target]
    assert os.path.exists(os.path.join(input_dir, file_name))
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    atom_num_orbital, orbital_types = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat'),
                                                         return_orbital_types=True)
    nsite = len(atom_num_orbital)
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f)
        spinful = info_dict["isspinful"]
    fid_H = h5py.File(os.path.join(input_dir, file_name), 'r')
    fid_rc = h5py.File(os.path.join(input_dir, 'rc.h5'), 'r')
    fid_rh = h5py.File(os.path.join(output_dir, prime_file_name), 'w')
    assert '[0, 0, 0, 1, 1]' in fid_H.keys()
    h5_dtype = fid_H['[0, 0, 0, 1, 1]'].dtype
    torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[h5_dtype.type]
    rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real, torch_dtype_complex=torch_dtype_complex,
                           device=torch_device, spinful=spinful)

    for key_str, hamiltonian in fid_H.items():
        if key_str not in fid_rc:
            warnings.warn(f'Hamiltonian matrix block ({key_str}) do not have local coordinate')
            continue
        rotation_matrix = torch.tensor(fid_rc[key_str], dtype=torch_dtype_real, device=torch_device)
        key = json.loads(key_str)
        atom_i = key[3] - 1
        atom_j = key[4] - 1
        assert atom_i >= 0
        assert atom_i < nsite
        assert atom_j >= 0
        assert atom_j < nsite
        if target == 'hamiltonian':
            rotated_hamiltonian = rotate_kernel.rotate_openmx_H(torch.tensor(hamiltonian), rotation_matrix,
                                                                orbital_types[atom_i], orbital_types[atom_j])
        elif target == 'phiVdphi':
            rotated_hamiltonian = rotate_kernel.rotate_openmx_phiVdphi(torch.tensor(hamiltonian), rotation_matrix,
                                                                       orbital_types[atom_i], orbital_types[atom_j])
        fid_rh[key_str] = rotated_hamiltonian.numpy()

    fid_H.close()
    fid_rc.close()
    fid_rh.close()


def rotate_back(input_dir, output_dir, target='hamiltonian'):
    torch_device = torch.device('cpu')
    assert target in ['hamiltonian', 'phiVdphi']
    file_name = {
        'hamiltonian': 'hamiltonians_pred.h5',
        'phiVdphi': 'phiVdphi_pred.h5',
    }[target]
    prime_file_name = {
        'hamiltonian': 'rh_pred.h5',
        'phiVdphi': 'rphiVdphi_pred.h5',
    }[target]
    assert os.path.exists(os.path.join(input_dir, prime_file_name))
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    atom_num_orbital, orbital_types = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat'),
                                                         return_orbital_types=True)
    nsite = len(atom_num_orbital)
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f)
        spinful = info_dict["isspinful"]
    fid_rc = h5py.File(os.path.join(input_dir, 'rc.h5'), 'r')
    fid_rh = h5py.File(os.path.join(input_dir, prime_file_name), 'r')
    fid_H = h5py.File(os.path.join(output_dir, file_name), 'w')
    assert '[0, 0, 0, 1, 1]' in fid_rh.keys()
    h5_dtype = fid_rh['[0, 0, 0, 1, 1]'].dtype
    torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[h5_dtype.type]
    rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real, torch_dtype_complex=torch_dtype_complex,
                           device=torch_device, spinful=spinful)

    for key_str, rotated_hamiltonian in fid_rh.items():
        assert key_str in fid_rc
        rotation_matrix = torch.tensor(fid_rc[key_str], dtype=torch_dtype_real, device=torch_device).T
        key = json.loads(key_str)
        atom_i = key[3] - 1
        atom_j = key[4] - 1
        assert atom_i >= 0
        assert atom_i < nsite
        assert atom_j >= 0
        assert atom_j < nsite
        if target == 'hamiltonian':
            hamiltonian = rotate_kernel.rotate_openmx_H(torch.tensor(rotated_hamiltonian), rotation_matrix,
                                                        orbital_types[atom_i], orbital_types[atom_j])
        elif target == 'phiVdphi':
            hamiltonian = rotate_kernel.rotate_openmx_phiVdphi(torch.tensor(rotated_hamiltonian), rotation_matrix,
                                                               orbital_types[atom_i], orbital_types[atom_j])
        fid_H[key_str] = hamiltonian.numpy()

    fid_H.close()
    fid_rc.close()
    fid_rh.close()
