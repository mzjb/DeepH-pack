import json
import argparse
import h5py
import numpy as np
import os
from time import time
from scipy import linalg 
import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, default="./",
        help="path of rlat.dat, orbital_types.dat, site_positions.dat, hamiltonians_pred.h5, and overlaps.h5"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./",
        help="path of output openmx.Band"
    )
    parser.add_argument(
        "--config", type=str,
        help="config file in the format of JSON"
    )
    parser.add_argument(
        "--ill_project", type=bool,
        help="projects out the eigenvectors of the overlap matrix that correspond to eigenvalues smaller than ill_threshold",
        default=True
    )
    parser.add_argument(
        "--ill_threshold", type=float,
        help="threshold for ill_project",
        default=5e-4
    )
    parser.add_argument(
        "--multiprocessing", type=int,
        help="multiprocessing for band calculation",
        default=0
    )
    return parser.parse_args()

parsed_args = parse_commandline()

def _create_dict_h5(filename):
    fid = h5py.File(filename, "r")
    d_out = {}
    for key in fid.keys():
        data = np.array(fid[key])
        nk = tuple(map(int, key[1:-1].split(',')))
        # BS: 
        # the matrix do not need be transposed in Python, 
        # But the transpose should be done in Julia.
        d_out[nk] = data # np.transpose(data)
    fid.close()
    return d_out


ev2Hartree = 0.036749324533634074
Bohr2Ang = 0.529177249


def genlist(x):
    return np.linspace(x[0], x[1], int(x[2]))


def k_data2num_ks(kdata):
    return int(kdata.split()[0])


def k_data2kpath(kdata):
    return [float(x) for x in kdata.split()[1:7]]


def std_out_array(a):
    return ''.join([str(x) + ' ' for x in a])


default_dtype = np.complex128

print(parsed_args.config)
with open(parsed_args.config) as f:
    config = json.load(f)
calc_job = config["calc_job"]

if os.path.isfile(os.path.join(parsed_args.input_dir, "info.json")):
    with open(os.path.join(parsed_args.input_dir, "info.json")) as f:
        spinful = json.load(f)["isspinful"]
else:
    spinful = False

site_positions = np.loadtxt(os.path.join(parsed_args.input_dir, "site_positions.dat"))

if len(site_positions.shape) == 2:
    nsites = site_positions.shape[1]
else:
    nsites = 1
    # in case of single atom


with open(os.path.join(parsed_args.input_dir, "orbital_types.dat")) as f:
    site_norbits = np.zeros(nsites, dtype=int)
    orbital_types = []
    for index_site in range(nsites):
        orbital_type = list(map(int, f.readline().split()))
        orbital_types.append(orbital_type)
        site_norbits[index_site] = np.sum(np.array(orbital_type) * 2 + 1)
    norbits = np.sum(site_norbits)
    site_norbits_cumsum = np.cumsum(site_norbits)

rlat = np.loadtxt(os.path.join(parsed_args.input_dir, "rlat.dat")).T
# require transposition while reading rlat.dat in python


print("read h5")
begin_time = time()
hamiltonians_pred = _create_dict_h5(os.path.join(parsed_args.input_dir, "hamiltonians_pred.h5"))
overlaps = _create_dict_h5(os.path.join(parsed_args.input_dir, "overlaps.h5"))
print("Time for reading h5: ", time() - begin_time, "s")

H_R = {}
S_R = {}

print("construct Hamiltonian and overlap matrix in the real space")
begin_time = time()

# BS:
# this is for debug python and julia
# in julia, you can use 'sort(collect(keys(hamiltonians_pred)))'
# for key in dict(sorted(hamiltonians_pred.items())).keys():
for key in hamiltonians_pred.keys():

    hamiltonian_pred = hamiltonians_pred[key]

    if key in overlaps.keys():
        overlap = overlaps[key]
    else:
        overlap = np.zeros_like(hamiltonian_pred)
    if spinful:
        overlap = np.vstack((np.hstack((overlap, np.zeros_like(overlap))), np.hstack((np.zeros_like(overlap), overlap))))
    R = key[:3]
    atom_i = key[3] - 1
    atom_j = key[4] - 1

    assert (site_norbits[atom_i], site_norbits[atom_j]) == hamiltonian_pred.shape
    assert (site_norbits[atom_i], site_norbits[atom_j]) == overlap.shape

    if R not in H_R.keys():
        H_R[R] = np.zeros((norbits, norbits), dtype=default_dtype)
        S_R[R] = np.zeros((norbits, norbits), dtype=default_dtype)

    for block_matrix_i in range(1, site_norbits[atom_i]+1):
        for block_matrix_j in range(1, site_norbits[atom_j]+1):
            index_i = site_norbits_cumsum[atom_i] - site_norbits[atom_i] + block_matrix_i - 1
            index_j = site_norbits_cumsum[atom_j] - site_norbits[atom_j] + block_matrix_j - 1
            H_R[R][index_i, index_j] = hamiltonian_pred[block_matrix_i-1, block_matrix_j-1]
            S_R[R][index_i, index_j] = overlap[block_matrix_i-1, block_matrix_j-1]


print("Time for constructing Hamiltonian and overlap matrix in the real space: ", time() - begin_time, " s")

if calc_job == "band":
    fermi_level = config["fermi_level"]
    k_data = config["k_data"]
    ill_project = parsed_args.ill_project or ("ill_project" in config.keys() and config["ill_project"])
    ill_threshold = max(parsed_args.ill_threshold, config["ill_threshold"] if ("ill_threshold" in config.keys()) else 0.)
    multiprocessing = max(parsed_args.multiprocessing, config["multiprocessing"] if ("multiprocessing" in config.keys()) else 0)

    print("calculate bands")
    num_ks = [k_data2num_ks(k) for k in k_data]
    kpaths = [k_data2kpath(k) for k in k_data]

    egvals = np.zeros((norbits, sum(num_ks)))

    begin_time = time()
    idx_k = 0
    # calculate total k points
    total_num_ks = sum(num_ks)
    list_index_kpath= []
    list_index_kxyz=[]
    for i in range(len(num_ks)):
        list_index_kpath = list_index_kpath + ([i]*num_ks[i])
        list_index_kxyz.extend(range(num_ks[i]))

    def process_worker(k_point):
        """ calculate band 

        Args:
            k_point (int): the index of k point of all calculated k points

        Returns:
            json: {
                "k_point":k_point, 
                "egval" (np array 1D) : eigen value , 
                "num_projected_out" (int) :  ill-conditioned eigenvalues detected。 default is 0
                }
        """
        index_kpath = list_index_kpath[k_point]
        kpath = kpaths[index_kpath]
        pnkpts = num_ks[index_kpath]
        kx = np.linspace(kpath[0], kpath[3], pnkpts)[list_index_kxyz[k_point]]
        ky = np.linspace(kpath[1], kpath[4], pnkpts)[list_index_kxyz[k_point]]
        kz = np.linspace(kpath[2], kpath[5], pnkpts)[list_index_kxyz[k_point]]

        H_k = np.matrix(np.zeros((norbits, norbits), dtype=default_dtype))
        S_k = np.matrix(np.zeros((norbits, norbits), dtype=default_dtype))
        for R in H_R.keys():
            H_k += H_R[R] * np.exp(1j*2*np.pi*np.dot([kx, ky, kz], R))
            S_k += S_R[R] * np.exp(1j*2*np.pi*np.dot([kx, ky, kz], R))
            # print(H_k)
        H_k = (H_k + H_k.getH())/2.
        S_k = (S_k + S_k.getH())/2.
        num_projected_out = 0
        if ill_project:
            egval_S, egvec_S = linalg.eig(S_k)
            project_index = np.argwhere(abs(egval_S)> ill_threshold)
            if len(project_index) != norbits:
                egvec_S = np.matrix(egvec_S[:, project_index])
                num_projected_out = norbits - len(project_index)
                H_k = egvec_S.H @ H_k @ egvec_S
                S_k = egvec_S.H @ S_k @ egvec_S
                egval = linalg.eigvalsh(H_k, S_k, lower=False)
                egval = np.concatenate([egval, np.full(num_projected_out, 1e4)])
            else:
                egval = linalg.eigvalsh(H_k, S_k, lower=False)
        else:
            #---------------------------------------------
            # BS: only eigenvalues are needed in this part, 
            # the upper matrix is used
            egval = linalg.eigvalsh(H_k, S_k, lower=False) 

        return {"k_point":k_point, "egval":egval, "num_projected_out":num_projected_out}
    
    # parallizing the band calculation
    if multiprocessing == 0:
        print(f'No use of multiprocessing')
        data_list = [process_worker(k_point) for k_point in tqdm.tqdm(range(sum(num_ks)))]
    else:
        pool_dict = {} if multiprocessing < 0 else {'nodes': multiprocessing}

        with Pool(**pool_dict) as pool:
            nodes = pool.nodes
            print(f'Use multiprocessing x {multiprocessing})')
            data_list = list(tqdm.tqdm(pool.imap(process_worker, range(sum(num_ks))), total=sum(num_ks)))
    
    # post-process returned band data, and store them in egvals with the order k_point
    projected_out = []
    for data in data_list:
        egvals[:, data["k_point"]] = data["egval"]
        if data["num_projected_out"] > 0:
            projected_out.append(data["num_projected_out"])
    if len(projected_out) > 0:
        print(f"There are {len(projected_out)} bands with ill-conditioned eigenvalues detected.")
        print(f"Projected out {int(np.average(projected_out))} eigenvalues on average.")
    print('Finish the calculation of %d k-points, have cost %d seconds' % (sum(num_ks), time() - begin_time))


    # output in openmx band format
    with open(os.path.join(parsed_args.output_dir, "openmx.Band"), "w") as f:
        f.write("{} {} {}\n".format(norbits, 0, ev2Hartree * fermi_level))
        openmx_rlat = np.reshape((rlat * Bohr2Ang), (1, -1))[0]
        f.write(std_out_array(openmx_rlat) + "\n")
        f.write(str(len(k_data)) + "\n")
        for line in k_data:
            f.write(line + "\n")
        idx_k = 0
        for i in range(len(kpaths)):
            pnkpts = num_ks[i]
            kstart = kpaths[i][:3]
            kend = kpaths[i][3:]
            k_list = np.zeros((pnkpts, 3))
            for alpha in range(3):
                k_list[:, alpha] = genlist([kstart[alpha], kend[alpha], pnkpts])
            for j in range(pnkpts):
                kvec = k_list[j, :]
                f.write("{} {}\n".format(norbits, std_out_array(kvec)))
                f.write(std_out_array(ev2Hartree * egvals[:, idx_k]) + "\n")
                idx_k += 1
