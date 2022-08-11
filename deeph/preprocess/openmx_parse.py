import os
import json
from math import pi

import tqdm
import argparse
import h5py
import numpy as np
from pymatgen.core.structure import Structure

from .abacus_get_data import periodic_table

Hartree2Ev = 27.2113845
Ev2Kcalmol = 23.061
Bohr2R = 0.529177249


def openmx_force_intferface(out_file_dir, save_dir=None, return_Etot=False, return_force=False):
    with open(out_file_dir, 'r') as out_file:
        lines = out_file.readlines()
        for index_line, line in enumerate(lines):
            if line.find('Total energy (Hartree) at MD = 1') != -1:
                assert lines[index_line + 3].find("Uele.") != -1
                assert lines[index_line + 5].find("Ukin.") != -1
                assert lines[index_line + 7].find("UH1.") != -1
                assert lines[index_line + 8].find("Una.") != -1
                assert lines[index_line + 9].find("Unl.") != -1
                assert lines[index_line + 10].find("Uxc0.") != -1
                assert lines[index_line + 20].find("Utot.") != -1
                parse_E = lambda x: float(x.split()[-1])
                E_tot = parse_E(lines[index_line + 20]) * Hartree2Ev
                E_kin = parse_E(lines[index_line + 5]) * Hartree2Ev
                E_delta_ee = parse_E(lines[index_line + 7]) * Hartree2Ev
                E_NA = parse_E(lines[index_line + 8]) * Hartree2Ev
                E_NL = parse_E(lines[index_line + 9]) * Hartree2Ev
                E_xc = parse_E(lines[index_line + 10]) * 2 * Hartree2Ev
                if save_dir is not None:
                    with open(os.path.join(save_dir, "openmx_E.json"), 'w') as E_file:
                        json.dump({
                            "Total energy": E_tot,
                            "E_kin": E_kin,
                            "E_delta_ee": E_delta_ee,
                            "E_NA": E_NA,
                            "E_NL": E_NL,
                            "E_xc": E_xc
                        }, E_file)
            if line.find('xyz-coordinates (Ang) and forces (Hartree/Bohr)') != -1:
                assert lines[index_line + 4].find("<coordinates.forces") != -1
                num_atom = int(lines[index_line + 5])
                forces = np.zeros((num_atom, 3))
                for index_atom in range(num_atom):
                    forces[index_atom] = list(
                        map(lambda x: float(x) * Hartree2Ev / Bohr2R, lines[index_line + 6 + index_atom].split()[-3:]))
                break
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "openmx_forces.dat"), forces)
    ret = (E_kin, E_delta_ee, E_NA, E_NL, E_xc)
    if return_Etot is True:
        ret = ret + (E_tot,)
    if return_force is True:
        ret = ret + (forces,)
    return ret


def openmx_parse_overlap(OLP_dir, output_dir):
    assert os.path.exists(os.path.join(OLP_dir, "output", "overlaps_0.h5")), "No overlap files found"
    assert os.path.exists(os.path.join(OLP_dir, "openmx.out")), "openmx.out not found"

    overlaps = read_non_parallel_hdf5('overlaps', os.path.join(OLP_dir, 'output'))
    assert len(overlaps.keys()) != 0, 'Can not found any overlap file'
    fid = h5py.File(os.path.join(output_dir, 'overlaps.h5'), 'w')
    for key_str, v in overlaps.items():
        fid[key_str] = v
    fid.close()

    orbital2l = {"s": 0, "p": 1, "d": 2, "f": 3}
    # parse openmx.out
    with open(os.path.join(OLP_dir, "openmx.out"), "r") as f:
        lines = f.readlines()
    orbital_dict = {}
    lattice = np.zeros((3, 3))
    frac_coords = []
    atomic_elements_str = []
    flag_read_orbital = False
    flag_read_lattice = False
    for index_line, line in enumerate(lines):
        if line.find('Definition.of.Atomic.Species>') != -1:
            flag_read_orbital = False
        if flag_read_orbital:
            element = line.split()[0]
            orbital_str = (line.split()[1]).split('-')[-1]
            l_list = []
            assert len(orbital_str) % 2 == 0
            for index_str in range(len(orbital_str) // 2):
                l_list.extend([orbital2l[orbital_str[index_str * 2]]] * int(orbital_str[index_str * 2 + 1]))
            orbital_dict[element] = l_list
        if line.find('<Definition.of.Atomic.Species') != -1:
            flag_read_orbital = True

        if line.find('Atoms.UnitVectors.Unit') != -1:
            assert line.split()[1] == "Ang", "Unit of lattice vector is not Angstrom"
            assert lines[index_line + 1].find("<Atoms.UnitVectors") != -1
            lattice[0, :] = np.array(list(map(float, lines[index_line + 2].split())))
            lattice[1, :] = np.array(list(map(float, lines[index_line + 3].split())))
            lattice[2, :] = np.array(list(map(float, lines[index_line + 4].split())))
            flag_read_lattice = True

        if line.find('Fractional coordinates of the final structure') != -1:
            index_atom = 0
            while (index_line + index_atom + 4) < len(lines):
                index_atom += 1
                line_split = lines[index_line + index_atom + 3].split()
                if len(line_split) == 0:
                    break
                assert len(line_split) == 5
                assert line_split[0] == str(index_atom)
                atomic_elements_str.append(line_split[1])
                frac_coords.append(np.array(list(map(float, line_split[2:]))))
    print("Found", len(frac_coords), "atoms")
    if flag_read_lattice is False:
        raise RuntimeError("Could not find lattice vector in openmx.out")
    if len(orbital_dict) == 0:
        raise RuntimeError("Could not find orbital information in openmx.out")
    frac_coords = np.array(frac_coords)
    cart_coords = frac_coords @ lattice

    np.savetxt(os.path.join(output_dir, "site_positions.dat"), cart_coords.T)
    np.savetxt(os.path.join(output_dir, "lat.dat"), lattice.T)
    np.savetxt(os.path.join(output_dir, "rlat.dat"), np.linalg.inv(lattice) * 2 * pi)
    np.savetxt(os.path.join(output_dir, "element.dat"),
               np.array(list(map(lambda x: periodic_table[x], atomic_elements_str))), fmt='%d')
    with open(os.path.join(output_dir, 'orbital_types.dat'), 'w') as orbital_types_f:
        for element_str in atomic_elements_str:
            for index_l, l in enumerate(orbital_dict[element_str]):
                if index_l == 0:
                    orbital_types_f.write(str(l))
                else:
                    orbital_types_f.write(f"  {l}")
            orbital_types_f.write('\n')


def read_non_parallel_hdf5(name, file_dir, num_p=256):
    Os = {}
    for index_p in range(num_p):
        if os.path.exists(os.path.join(file_dir, f"{name}_{index_p}.h5")):
            fid = h5py.File(os.path.join(file_dir, f"{name}_{index_p}.h5"), 'r')
            for key_str, O_nm in fid.items():
                Os[key_str] = O_nm[...]
    assert not os.path.exists(os.path.join(file_dir, f"{name}_{num_p}.h5")), "Increase num_p because some overlap files are missing"
    return Os


def read_hdf5(name, file_dir):
    Os = {}
    fid = h5py.File(os.path.join(file_dir, f"{name}.h5"), 'r')
    for key_str, O_nm in fid.items():
        Os[key_str] = O_nm[...]
    return Os


class OijLoad:
    def __init__(self, output_dir):
        print("get data from:", output_dir)
        self.if_load_scfout = False
        self.output_dir = output_dir
        term_non_parallel_list = ['H', 'T', 'V_xc', 'O_xc', 'O_dVHart', 'O_NA', 'O_NL', 'Rho']
        self.term_h5_dict = {}
        for term in term_non_parallel_list:
            self.term_h5_dict[term] = read_non_parallel_hdf5(term, output_dir)

        self.term_h5_dict['H_add'] = {}
        for key_str in self.term_h5_dict['T'].keys():
            tmp = np.zeros_like(self.term_h5_dict['T'][key_str])
            for term in ['T', 'V_xc', 'O_dVHart', 'O_NA', 'O_NL']:
                tmp += self.term_h5_dict[term][key_str]
            self.term_h5_dict['H_add'][key_str] = tmp

        self.dig_term = {}
        for term in ['E_dVHart_a', 'E_xc_pcc']:
            self.dig_term[term] = np.loadtxt(os.path.join(output_dir, f'{term}.dat'))

    def cal_Eij(self):
        term_list = ["E_kin", "E_NA", "E_NL", "E_delta_ee", "E_xc"]
        self.Eij = {term: {} for term in term_list}
        self.R_list = []
        for key_str in self.term_h5_dict['T'].keys():
            key = json.loads(key_str)
            R = (key[0], key[1], key[2])
            if R not in self.R_list:
                self.R_list.append(R)
            atom_i = key[3] - 1
            atom_j = key[4] - 1

            self.Eij["E_NA"][key_str] = (self.term_h5_dict["O_NA"][key_str] * self.term_h5_dict["Rho"][key_str]).sum() * 2
            self.Eij["E_NL"][key_str] = (self.term_h5_dict["O_NL"][key_str] * self.term_h5_dict["Rho"][key_str]).sum() * 2
            self.Eij["E_kin"][key_str] = (self.term_h5_dict["T"][key_str] * self.term_h5_dict["Rho"][key_str]).sum() * 2
            self.Eij["E_delta_ee"][key_str] = (self.term_h5_dict["O_dVHart"][key_str] * self.term_h5_dict["Rho"][key_str]).sum()
            self.Eij["E_xc"][key_str] = (self.term_h5_dict["O_xc"][key_str] * self.term_h5_dict["Rho"][key_str]).sum() * 2
            if (atom_i == atom_j) and (R == (0, 0, 0)):
                self.Eij["E_delta_ee"][key_str] -= self.dig_term['E_dVHart_a'][atom_i]
                self.Eij["E_xc"][key_str] += self.dig_term['E_xc_pcc'][atom_i] * 2

    def load_scfout(self):
        self.if_load_scfout = True
        term_list = ["hamiltonians", "overlaps", "density_matrixs"]
        default_dtype = np.complex128

        for term in term_list:
            self.term_h5_dict[term] = read_hdf5(term, self.output_dir)

        site_positions = np.loadtxt(os.path.join(self.output_dir, 'site_positions.dat')).T
        self.lat = np.loadtxt(os.path.join(self.output_dir, 'lat.dat')).T
        self.rlat = np.loadtxt(os.path.join(self.output_dir, 'rlat.dat')).T
        nsites = site_positions.shape[0]

        self.orbital_types = []
        with open(os.path.join(self.output_dir, 'orbital_types.dat'), 'r') as orbital_types_f:
            for index_site in range(nsites):
                self.orbital_types.append(np.array(list(map(int, orbital_types_f.readline().split()))))
        site_norbits = list(map(lambda x: (2 * x + 1).sum(), self.orbital_types))
        site_norbits_cumsum = np.cumsum(site_norbits)
        norbits = sum(site_norbits)

        self.term_R_dict = {term: {} for term in self.term_h5_dict.keys()}
        for key_str in tqdm.tqdm(self.term_h5_dict['overlaps'].keys()):
            key = json.loads(key_str)
            R = (key[0], key[1], key[2])
            atom_i = key[3] - 1
            atom_j = key[4] - 1
            if R not in self.term_R_dict['overlaps']:
                for term_R in self.term_R_dict.values():
                    term_R[R] = np.zeros((norbits, norbits), dtype=default_dtype)
            matrix_slice_i = slice(site_norbits_cumsum[atom_i] - site_norbits[atom_i], site_norbits_cumsum[atom_i])
            matrix_slice_j = slice(site_norbits_cumsum[atom_j] - site_norbits[atom_j], site_norbits_cumsum[atom_j])
            for term, term_R in self.term_R_dict.items():
                term_R[R][matrix_slice_i, matrix_slice_j] = np.array(self.term_h5_dict[term][key_str]).astype(
                    dtype=default_dtype)

    def get_E_band(self):
        E_band = 0.0
        for R in self.term_R_dict['T'].keys():
            E_band += (self.term_R_dict['density_matrixs'][R] * self.term_R_dict['H_add'][R]).sum()
        return E_band

    def get_E_band2(self):
        E_band = 0.0
        for R in self.term_R_dict['T'].keys():
            E_band += (self.term_R_dict['density_matrixs'][R] * self.term_R_dict['hamiltonians'][R]).sum()
        return E_band

    def get_E_band3(self):
        E_band = 0.0
        for R in self.term_R_dict['T'].keys():
            E_band += (self.term_R_dict['density_matrixs'][R] * self.term_R_dict['H'][R]).sum()
        return E_band

    def sum_Eij(self, term):
        ret = 0.0
        for value in self.Eij[term].values():
            ret += value
        return ret

    def get_E_NL(self):
        assert self.if_load_scfout == True
        E_NL = 0.0
        for R in self.term_R_dict['T'].keys():
            E_NL += (self.term_R_dict['density_matrixs'][R] * self.term_R_dict['O_NL'][R]).sum()
        return E_NL

    def save_Vij(self, save_dir):
        for term, h5_file_name in zip(["O_NA", "O_dVHart", "V_xc", "H_add", "Rho"],
                                      ["V_nas", "V_delta_ees", "V_xcs", "hamiltonians", "density_matrixs"]):
            fid = h5py.File(os.path.join(save_dir, f'{h5_file_name}.h5'), "w")
            for k, v in self.term_h5_dict[term].items():
                fid[k] = v
            fid.close()

    def get_E5ij(self):
        term_list = ["E_kin", "E_NA", "E_NL", "E_delta_ee", "E_xc"]
        E_dict = {term: 0 for term in term_list}
        E5ij = {}
        for key_str in self.Eij[term_list[0]].keys():
            tmp = 0.0
            for term in term_list:
                v = self.Eij[term][key_str]
                E_dict[term] += v
                tmp += v
            if key_str in E5ij:
                E5ij[key_str] += tmp
            else:
                E5ij[key_str] = tmp
        return E5ij, E_dict

    def save_Eij(self, save_dir):
        fid_tmp, E_dict = self.get_E5ij()

        fid = h5py.File(os.path.join(save_dir, f'E_ij.h5'), "w")
        for k, v in fid_tmp.items():
            fid[k] = v
        fid.close()

        with open(os.path.join(save_dir, "openmx_E_ij_E.json"), 'w') as E_file:
            json.dump({
                "E_kin": E_dict["E_kin"],
                "E_delta_ee": E_dict["E_delta_ee"],
                "E_NA": E_dict["E_NA"],
                "E_NL": E_dict["E_NL"],
                "E_xc": E_dict["E_xc"]
            }, E_file)

        # return E_dict["E_delta_ee"], E_dict["E_xc"]
        return E_dict["E_kin"], E_dict["E_delta_ee"], E_dict["E_NA"], E_dict["E_NL"], E_dict["E_xc"]

    def get_E5i(self):
        term_list = ["E_kin", "E_NA", "E_NL", "E_delta_ee", "E_xc"]
        E_dict = {term: 0 for term in term_list}
        E5i = {}
        for key_str in self.Eij[term_list[0]].keys():
            key = json.loads(key_str)
            atom_i_str = str(key[3] - 1)
            tmp = 0.0
            for term in term_list:
                v = self.Eij[term][key_str]
                E_dict[term] += v
                tmp += v
            if atom_i_str in E5i:
                E5i[atom_i_str] += tmp
            else:
                E5i[atom_i_str] = tmp
        return E5i, E_dict

    def save_Ei(self, save_dir):
        fid_tmp, E_dict = self.get_E5i()

        fid = h5py.File(os.path.join(save_dir, f'E_i.h5'), "w")
        for k, v in fid_tmp.items():
            fid[k] = v
        fid.close()
        with open(os.path.join(save_dir, "openmx_E_i_E.json"), 'w') as E_file:
            json.dump({
                "E_kin": E_dict["E_kin"],
                "E_delta_ee": E_dict["E_delta_ee"],
                "E_NA": E_dict["E_NA"],
                "E_NL": E_dict["E_NL"],
                "E_xc": E_dict["E_xc"]
            }, E_file)
        return E_dict["E_kin"], E_dict["E_delta_ee"], E_dict["E_NA"], E_dict["E_NL"], E_dict["E_xc"]

    def get_R_list(self):
        return self.R_list


class GetEEiEij:
    def __init__(self, input_dir):
        self.load_kernel = OijLoad(os.path.join(input_dir, "output"))
        self.E_kin, self.E_delta_ee, self.E_NA, self.E_NL, self.E_xc, self.Etot, self.force = openmx_force_intferface(
            os.path.join(input_dir, "openmx.out"), save_dir=None, return_Etot=True, return_force=True)
        self.load_kernel.cal_Eij()

    def get_Etot(self):
        # unit: kcal mol^-1
        return self.Etot * Ev2Kcalmol

    def get_force(self):
        # unit: kcal mol^-1 Angstrom^-1
        return self.force * Ev2Kcalmol

    def get_E5(self):
        # unit: kcal mol^-1
        return (self.E_kin + self.E_delta_ee + self.E_NA + self.E_NL + self.E_xc) * Ev2Kcalmol

    def get_E5i(self):
        # unit: kcal mol^-1
        E5i, E_from_i_dict = self.load_kernel.get_E5i()
        assert np.allclose(self.E_kin, E_from_i_dict["E_kin"])
        assert np.allclose(self.E_delta_ee, E_from_i_dict["E_delta_ee"])
        assert np.allclose(self.E_NA, E_from_i_dict["E_NA"])
        assert np.allclose(self.E_NL, E_from_i_dict["E_NL"])
        assert np.allclose(self.E_xc, E_from_i_dict["E_xc"], rtol=1.e-3)
        return {k: v * Ev2Kcalmol for k, v in E5i.items()}

    def get_E5ij(self):
        # unit: kcal mol^-1
        E5ij, E_from_ij_dict = self.load_kernel.get_E5ij()
        assert np.allclose(self.E_kin, E_from_ij_dict["E_kin"])
        assert np.allclose(self.E_delta_ee, E_from_ij_dict["E_delta_ee"])
        assert np.allclose(self.E_NA, E_from_ij_dict["E_NA"])
        assert np.allclose(self.E_NL, E_from_ij_dict["E_NL"])
        assert np.allclose(self.E_xc, E_from_ij_dict["E_xc"], rtol=1.e-3)
        return {k: v * Ev2Kcalmol for k, v in E5ij.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Hamiltonian')
    parser.add_argument(
        '--input_dir', type=str, default='./',
        help='path of openmx.out, and output'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./',
        help='path of output E_xc_ij.h5, E_delta_ee_ij.h5, site_positions.dat, lat.dat, element.dat, and R_list.dat'
    )
    parser.add_argument('--Ei', action='store_true')
    parser.add_argument('--stru_dir', type=str, default='POSCAR', help='path of structure file')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_kernel = OijLoad(os.path.join(args.input_dir, "output"))
    E_kin, E_delta_ee, E_NA, E_NL, E_xc = openmx_force_intferface(os.path.join(args.input_dir, "openmx.out"), args.output_dir)
    load_kernel.cal_Eij()
    if args.Ei:
        E_kin_from_ij, E_delta_ee_from_ij, E_NA_from_ij, E_NL_from_ij, E_xc_from_ij = load_kernel.save_Ei(args.output_dir)
    else:
        E_kin_from_ij, E_delta_ee_from_ij, E_NA_from_ij, E_NL_from_ij, E_xc_from_ij = load_kernel.save_Eij(args.output_dir)
    assert np.allclose(E_kin, E_kin_from_ij)
    assert np.allclose(E_delta_ee, E_delta_ee_from_ij)
    assert np.allclose(E_NA, E_NA_from_ij)
    assert np.allclose(E_NL, E_NL_from_ij)
    assert np.allclose(E_xc, E_xc_from_ij, rtol=1.e-3)

    structure = Structure.from_file(args.stru_dir)
    np.savetxt(os.path.join(args.output_dir, "site_positions.dat"), structure.cart_coords.T)
    np.savetxt(os.path.join(args.output_dir, "lat.dat"), structure.lattice.matrix.T)
    np.savetxt(os.path.join(args.output_dir, "element.dat"), structure.atomic_numbers, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, "R_list.dat"), load_kernel.get_R_list(), fmt='%d')
