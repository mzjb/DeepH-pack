# Script for interface from ABACUS (http://abacus.ustc.edu.cn/) to DeepH-pack
# Coded by ZC Tang @ Tsinghua Univ. e-mail: az_txycha@126.com
# Modified by He Li @ Tsinghua Univ. & XY Zhou @ Peking Univ.
# To use this script, please add 'out_mat_hs2    1' in ABACUS INPUT File
# Current version is capable of coping with f-orbitals
# 20220717: Read structure from running_scf.log
# 20220919: The suffix of the output sub-directories (OUT.suffix) can be set by ["basic"]["abacus_suffix"] keyword in preprocess.ini
# 20220920: Supporting cartesian coordinates in the log file

import os
import sys
import json
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import argparse
import h5py


Bohr2Ang = 0.529177249
periodic_table = {'Ac': 89, 'Ag': 47, 'Al': 13, 'Am': 95, 'Ar': 18, 'As': 33, 'At': 85, 'Au': 79, 'B': 5, 'Ba': 56,
                  'Be': 4, 'Bi': 83, 'Bk': 97, 'Br': 35, 'C': 6, 'Ca': 20, 'Cd': 48, 'Ce': 58, 'Cf': 98, 'Cl': 17,
                  'Cm': 96, 'Co': 27, 'Cr': 24, 'Cs': 55, 'Cu': 29, 'Dy': 66, 'Er': 68, 'Es': 99, 'Eu': 63, 'F': 9,
                  'Fe': 26, 'Fm': 100, 'Fr': 87, 'Ga': 31, 'Gd': 64, 'Ge': 32, 'H': 1, 'He': 2, 'Hf': 72, 'Hg': 80,
                  'Ho': 67, 'I': 53, 'In': 49, 'Ir': 77, 'K': 19, 'Kr': 36, 'La': 57, 'Li': 3, 'Lr': 103, 'Lu': 71,
                  'Md': 101, 'Mg': 12, 'Mn': 25, 'Mo': 42, 'N': 7, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10, 'Ni': 28,
                  'No': 102, 'Np': 93, 'O': 8, 'Os': 76, 'P': 15, 'Pa': 91, 'Pb': 82, 'Pd': 46, 'Pm': 61, 'Po': 84,
                  'Pr': 59, 'Pt': 78, 'Pu': 94, 'Ra': 88, 'Rb': 37, 'Re': 75, 'Rh': 45, 'Rn': 86, 'Ru': 44, 'S': 16,
                  'Sb': 51, 'Sc': 21, 'Se': 34, 'Si': 14, 'Sm': 62, 'Sn': 50, 'Sr': 38, 'Ta': 73, 'Tb': 65, 'Tc': 43,
                  'Te': 52, 'Th': 90, 'Ti': 22, 'Tl': 81, 'Tm': 69, 'U': 92, 'V': 23, 'W': 74, 'Xe': 54, 'Y': 39,
                  'Yb': 70, 'Zn': 30, 'Zr': 40, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                  'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


class OrbAbacus2DeepH:
    def __init__(self):
        self.Us_abacus2deeph = {}
        self.Us_abacus2deeph[0] = np.eye(1)
        self.Us_abacus2deeph[1] = np.eye(3)[[1, 2, 0]]
        self.Us_abacus2deeph[2] = np.eye(5)[[0, 3, 4, 1, 2]]
        self.Us_abacus2deeph[3] = np.eye(7)[[0, 1, 2, 3, 4, 5, 6]]

        minus_dict = {
            1: [0, 1],
            2: [3, 4],
            3: [1, 2, 5, 6],
        }
        for k, v in minus_dict.items():
            self.Us_abacus2deeph[k][v] *= -1

    def get_U(self, l):
        if l > 3:
            raise NotImplementedError("Only support l = s, p, d, f")
        return self.Us_abacus2deeph[l]

    def transform(self, mat, l_lefts, l_rights):
        block_lefts = block_diag(*[self.get_U(l_left) for l_left in l_lefts])
        block_rights = block_diag(*[self.get_U(l_right) for l_right in l_rights])
        return block_lefts @ mat @ block_rights.T

def abacus_parse(input_path, output_path, data_name, only_S=False, get_r=False):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    def find_target_line(f, target):
        line = f.readline()
        while line:
            if target in line:
                return line
            line = f.readline()
        return None
    if only_S:
        log_file_name = "running_get_S.log"
    else:
        log_file_name = "running_scf.log"
    with open(os.path.join(input_path, data_name, log_file_name), 'r') as f:
        f.readline()
        line = f.readline()
        # assert "WELCOME TO ABACUS" in line
        assert find_target_line(f, "READING UNITCELL INFORMATION") is not None, 'Cannot find "READING UNITCELL INFORMATION" in log file'
        num_atom_type = int(f.readline().split()[-1])

        assert find_target_line(f, "lattice constant (Bohr)") is not None
        lattice_constant = float(f.readline().split()[-1]) # unit is Angstrom

        site_norbits_dict = {}
        orbital_types_dict = {}
        for index_type in range(num_atom_type):
            tmp = find_target_line(f, "READING ATOM TYPE")
            assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
            assert tmp.split()[-1] == str(index_type + 1)
            if tmp is None:
                raise Exception(f"Cannot find ATOM {index_type} in {log_file_name}")

            line = f.readline()
            assert "atom label =" in line
            atom_label = line.split()[-1]
            assert atom_label in periodic_table, "Atom label should be in periodic table"
            atom_type = periodic_table[atom_label]

            current_site_norbits = 0
            current_orbital_types = []
            while True:
                line = f.readline()
                if "number of zeta" in line:
                    tmp = line.split()
                    L = int(tmp[0][2:-1])
                    num_L = int(tmp[-1])
                    current_site_norbits += (2 * L + 1) * num_L
                    current_orbital_types.extend([L] * num_L)
                else:
                    break
            site_norbits_dict[atom_type] = current_site_norbits
            orbital_types_dict[atom_type] = current_orbital_types

        line = find_target_line(f, "TOTAL ATOM NUMBER")
        assert line is not None, 'Cannot find "TOTAL ATOM NUMBER" in log file'
        nsites = int(line.split()[-1])
        
        line = find_target_line(f, " COORDINATES")
        assert line is not None, 'Cannot find "DIRECT COORDINATES" or "CARTESIAN COORDINATES" in log file'
        if "DIRECT" in line:
            coords_type = "direct" 
        elif "CARTESIAN" in line:
            coords_type = "cartesian" 
        else:
            raise ValueError('Cannot find "DIRECT COORDINATES" or "CARTESIAN COORDINATES" in log file')

        assert "atom" in f.readline()
        frac_coords = np.zeros((nsites, 3))
        site_norbits = np.zeros(nsites, dtype=int)
        element = np.zeros(nsites, dtype=int)
        for index_site in range(nsites):
            line = f.readline()
            tmp = line.split()
            assert "tau" in tmp[0]
            atom_label = ''.join(re.findall(r'[A-Za-z]', tmp[0][5:]))
            assert atom_label in periodic_table, "Atom label should be in periodic table"
            element[index_site] = periodic_table[atom_label]
            site_norbits[index_site] = site_norbits_dict[element[index_site]]
            frac_coords[index_site, :] = np.array(tmp[1:4])
        norbits = int(np.sum(site_norbits))
        site_norbits_cumsum = np.cumsum(site_norbits)

        assert find_target_line(f, "Lattice vectors: (Cartesian coordinate: in unit of a_0)") is not None
        lattice = np.zeros((3, 3))
        for index_lat in range(3):
            lattice[index_lat, :] = np.array(f.readline().split())
        if coords_type == "cartesian":
            frac_coords = frac_coords @ np.matrix(lattice).I
        lattice = lattice * lattice_constant

        if only_S:
            spinful = False
        else:
            line = find_target_line(f, "NSPIN")
            assert line is not None, 'Cannot find "NSPIN" in log file'
            if "NSPIN == 1" in line:
                spinful = False
            elif "NSPIN == 4" in line:
                spinful = True
            else:
                raise ValueError(f'{line} is not supported')

        if only_S:
            fermi_level = 0.0
        else:
            line = find_target_line(f, "EFERMI")
            assert line is not None, 'Cannot find "EFERMI" in log file'
            assert "eV" in line
            fermi_level = float(line.split()[2])
            assert find_target_line(f, "EFERMI") is None, "There is more than one EFERMI in log file"

    np.savetxt(os.path.join(output_path, "lat.dat"), np.transpose(lattice))
    np.savetxt(os.path.join(output_path, "rlat.dat"), np.linalg.inv(lattice) * 2 * np.pi)
    cart_coords = frac_coords @ lattice
    np.savetxt(os.path.join(output_path, "site_positions.dat").format(output_path), np.transpose(cart_coords))
    np.savetxt(os.path.join(output_path, "element.dat"), element, fmt='%d')
    info = {'nsites' : nsites, 'isorthogonal': False, 'isspinful': spinful, 'norbits': norbits, 'fermi_level': fermi_level}
    with open('{}/info.json'.format(output_path), 'w') as info_f:
        json.dump(info, info_f)
    with open(os.path.join(output_path, "orbital_types.dat"), 'w') as f:
        for atomic_number in element:
            for index_l, l in enumerate(orbital_types_dict[atomic_number]):
                if index_l == 0:
                    f.write(str(l))
                else:
                    f.write(f"  {l}")
            f.write('\n')

    U_orbital = OrbAbacus2DeepH()
    def parse_matrix(matrix_path, factor, spinful=False):
        matrix_dict = dict()
        with open(matrix_path, 'r') as f:
            line = f.readline() # read "Matrix Dimension of ..."
            if not "Matrix Dimension of" in line:
                line = f.readline() # ABACUS >= 3.0
                assert "Matrix Dimension of" in line
            f.readline() # read "Matrix number of ..."
            norbits = int(line.split()[-1])
            for line in f:
                line1 = line.split()
                if len(line1) == 0:
                    break
                num_element = int(line1[3])
                if num_element != 0:
                    R_cur = np.array(line1[:3]).astype(int)
                    line2 = f.readline().split()
                    line3 = f.readline().split()
                    line4 = f.readline().split()
                    if not spinful:
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(float), np.array(line3).astype(int),
                                                      np.array(line4).astype(int)), shape=(norbits, norbits)).toarray()
                    else:
                        line2 = np.char.replace(line2, '(', '')
                        line2 = np.char.replace(line2, ')', 'j')
                        line2 = np.char.replace(line2, ',', '+')
                        line2 = np.char.replace(line2, '+-', '-')
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.complex128), np.array(line3).astype(int),
                                                      np.array(line4).astype(int)), shape=(norbits, norbits)).toarray()
                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            key_str = f"[{R_cur[0]}, {R_cur[1]}, {R_cur[2]}, {index_site_i + 1}, {index_site_j + 1}]"
                            mat = hamiltonian_cur[(site_norbits_cumsum[index_site_i]
                                                  - site_norbits[index_site_i]) * (1 + spinful):
                                                  site_norbits_cumsum[index_site_i] * (1 + spinful),
                                  (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]) * (1 + spinful):
                                  site_norbits_cumsum[index_site_j] * (1 + spinful)]
                            if abs(mat).max() < 1e-8:
                                continue
                            if not spinful:
                                mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]],
                                                          orbital_types_dict[element[index_site_j]])
                            else:
                                mat = mat.reshape((site_norbits[index_site_i], 2, site_norbits[index_site_j], 2))
                                mat = mat.transpose((1, 0, 3, 2)).reshape((2 * site_norbits[index_site_i],
                                                                           2 * site_norbits[index_site_j]))
                                mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]] * 2,
                                                          orbital_types_dict[element[index_site_j]] * 2)
                            matrix_dict[key_str] = mat * factor
        return matrix_dict, norbits

    if only_S:
        overlap_dict, tmp = parse_matrix(os.path.join(input_path, "SR.csr"), 1)
        assert tmp == norbits
    else:
        hamiltonian_dict, tmp = parse_matrix(
            os.path.join(input_path, data_name, "data-HR-sparse_SPIN0.csr"), 13.605698, # Ryd2eV
            spinful=spinful)
        assert tmp == norbits * (1 + spinful)
        overlap_dict, tmp = parse_matrix(os.path.join(input_path, data_name, "data-SR-sparse_SPIN0.csr"), 1,
                                         spinful=spinful)
        assert tmp == norbits * (1 + spinful)
        if spinful:
            overlap_dict_spinless = {}
            for k, v in overlap_dict.items():
                overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
            overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

    if not only_S:
        with h5py.File(os.path.join(output_path, "hamiltonians.h5"), 'w') as fid:
            for key_str, value in hamiltonian_dict.items():
                fid[key_str] = value
    with h5py.File(os.path.join(output_path, "overlaps.h5"), 'w') as fid:
        for key_str, value in overlap_dict.items():
            fid[key_str] = value
    if get_r:
        def parse_r_matrix(matrix_path, factor):
            matrix_dict = dict()
            with open(matrix_path, 'r') as f:
                line = f.readline();
                norbits = int(line.split()[-1])
                for line in f:
                    line1 = line.split()
                    if len(line1) == 0:
                        break
                    assert len(line1) > 3
                    R_cur = np.array(line1[:3]).astype(int)
                    mat_cur = np.zeros((3, norbits * norbits))
                    for line_index in range(norbits * norbits):
                        line_mat = f.readline().split()
                        assert len(line_mat) == 3
                        mat_cur[:, line_index] = np.array(line_mat)
                    mat_cur = mat_cur.reshape((3, norbits, norbits))

                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            for direction in range(3):
                                key_str = f"[{R_cur[0]}, {R_cur[1]}, {R_cur[2]}, {index_site_i + 1}, {index_site_j + 1}, {direction + 1}]"
                                mat = mat_cur[direction, site_norbits_cumsum[index_site_i]
                                              - site_norbits[index_site_i]:site_norbits_cumsum[index_site_i],
                                      site_norbits_cumsum[index_site_j]
                                      - site_norbits[index_site_j]:site_norbits_cumsum[index_site_j]]
                                if abs(mat).max() < 1e-8:
                                    continue
                                mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]],
                                                          orbital_types_dict[element[index_site_j]])
                                matrix_dict[key_str] = mat * factor
            return matrix_dict, norbits
        position_dict, tmp = parse_r_matrix(os.path.join(input_path, data_name, "data-rR-tr_SPIN1"), 0.529177249) # Bohr2Ang
        assert tmp == norbits

        with h5py.File(os.path.join(output_path, "positions.h5"), 'w') as fid:
            for key_str, value in position_dict.items():
                fid[key_str] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Hamiltonian')
    parser.add_argument(
        '-i','--input_dir', type=str, default='./',
        help='path of output subdirectory'
        )
    parser.add_argument(
        '-o','--output_dir', type=str, default='./',
        help='path of output .h5 and .dat'
        )
    parser.add_argument(
        '-a','--abacus_suffix', type=str, default='ABACUS',
        help='suffix of output subdirectory'
        )
    parser.add_argument(
        '-S','--only_S', type=int, default=0
        )
    parser.add_argument(
        '-g','--get_r', type=int, default=0
        )
    args = parser.parse_args()

    input_path = args.input_dir
    output_path = args.output_dir
    data_name = "OUT." + args.abacus_suffix
    only_S = bool(args.only_S)
    get_r = bool(args.get_r)
    print("only_S: {}".format(only_S))
    print("get_r: {}".format(get_r))
    abacus_parse(input_path, output_path, data_name, only_S, get_r)
