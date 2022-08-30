import os
import numpy as np
from numpy.core.fromnumeric import sort
import scipy as sp
import h5py
import json
from scipy.io import FortranFile

# Transfer SIESTA output to DeepH format
# DeepH-pack: https://deeph-pack.readthedocs.io/en/latest/index.html
# Coded by ZC Tang @ Tsinghua Univ. e-mail: az_txycha@126.com

def siesta_parse(input_path, output_path):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # finds system name
    f_list = os.listdir(input_path)
    for f_name in f_list:
        if f_name[::-1][0:9] == 'XDNI_BRO.':
            system_name = f_name[:-9]

    with open('{}/{}.STRUCT_OUT'.format(input_path,system_name), 'r') as struct: # structure info from standard output
        lattice = np.empty((3,3))
        for i in range(3):
            line = struct.readline()
            linesplit = line.split()
            lattice[i,:] = linesplit[:]
        np.savetxt('{}/lat.dat'.format(output_path), np.transpose(lattice), fmt='%.18e') 
        line = struct.readline()
        linesplit = line.split()
        num_atoms = int(linesplit[0])
        atom_coord = np.empty((num_atoms, 4))
        for i in range(num_atoms):
            line = struct.readline()
            linesplit = line.split()
            atom_coord[i, :] = linesplit[1:]
        np.savetxt('{}/element.dat'.format(output_path), atom_coord[:,0], fmt='%d')

    atom_coord_cart = np.genfromtxt('{}/{}.XV'.format(input_path,system_name),skip_header = 4)
    atom_coord_cart = atom_coord_cart[:,2:5] * 0.529177249
    np.savetxt('{}/site_positions.dat'.format(output_path), np.transpose(atom_coord_cart))

    orb_indx = np.genfromtxt('{}/{}.ORB_INDX'.format(input_path,system_name), skip_header=3, skip_footer=17)
    # orb_indx rows: 0 orbital id   1 atom id   2 atom type   3 element symbol
    #                4 orbital id within atom   5 n           6 l
    #                7 m            8 zeta      9 Polarized? 10 orbital symmetry
    #               11 rc(a.u.)     12-14 R     15 equivalent orbital index in uc

    orb_indx[:,12:15]=orb_indx[:,12:15]

    with open('{}/R_list.dat'.format(output_path),'w') as R_list_f:
        R_prev = np.empty(3)
        for i in range(len(orb_indx)):
            R = orb_indx[i, 12:15]
            if (R != R_prev).any():
                R_prev = R
                R_list_f.write('{} {} {}\n'.format(int(R[0]), int(R[1]), int(R[2])))

    ia2Riua = np.empty((0,4)) #DeepH key
    ia = 0
    for i in range(len(orb_indx)):
        if orb_indx[i][1] != ia:
            ia = orb_indx[i][1]
            Riua = np.empty((1,4))
            Riua[0,0:3] = orb_indx[i][12:15]
            iuo = int(orb_indx[i][15])
            iua = int(orb_indx[iuo-1,1])
            Riua[0,3] = int(iua)
            ia2Riua = np.append(ia2Riua, Riua)
    ia2Riua = ia2Riua.reshape(int(len(ia2Riua)/4),4)


    #hamiltonians.h5, density_matrixs.h5, overlap.h5
    info = {'nsites' : num_atoms, 'isorthogonal': False, 'isspinful': False, 'norbits': len(orb_indx)}
    with open('{}/info.json'.format(output_path), 'w') as info_f:
        json.dump(info, info_f)

    a1 = lattice[0, :]
    a2 = lattice[1, :]
    a3 = lattice[2, :]
    b1 = 2 * np.pi * np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))
    b2 = 2 * np.pi * np.cross(a3, a1) / (np.dot(a2, np.cross(a3, a1)))
    b3 = 2 * np.pi * np.cross(a1, a2) / (np.dot(a3, np.cross(a1, a2)))
    rlattice = np.array([b1, b2, b3])
    np.savetxt('{}/rlat.dat'.format(output_path), np.transpose(rlattice), fmt='%.18e')

    # Cope with orbital type information
    i = 0
    with open('{}/orbital_types.dat'.format(output_path), 'w') as orb_type_f:
        atom_current = 0
        while True: # Loop over atoms in unitcell
            if atom_current != orb_indx[i, 1]:
                if atom_current != 0:
                    for j in range(4):
                        for _ in range(int(atom_orb_cnt[j]/(2*j+1))):
                            orb_type_f.write('{}  '.format(j))
                    orb_type_f.write('\n')

                atom_current = int(orb_indx[i, 1])
                atom_orb_cnt = np.array([0,0,0,0]) # number of s, p, d, f orbitals in specific atom
            l = int(orb_indx[i, 6])
            atom_orb_cnt[l] += 1
            i += 1
            if i > len(orb_indx)-1:
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j]/(2*j+1))):
                        orb_type_f.write('{}  '.format(j))
                orb_type_f.write('\n')
                break
            if orb_indx[i, 0] != orb_indx[i, 15]:
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j]/(2*j+1))):
                        orb_type_f.write('{}  '.format(j))
                orb_type_f.write('\n')
                break

    # yields key for *.h5 file
    orb2deephorb = np.zeros((len(orb_indx), 5))
    atom_current = 1
    orb_atom_current = np.empty((0)) # stores orbitals' id in siesta, n, l, m and z, will be reshaped into orb*5
    t = 0 
    for i in range(len(orb_indx)):  
        orb_atom_current = np.append(orb_atom_current, i)
        orb_atom_current = np.append(orb_atom_current, orb_indx[i,5:9])
        if i != len(orb_indx)-1 :
            if orb_indx[i+1,1] != atom_current:
                orb_atom_current = np.reshape(orb_atom_current,((int(len(orb_atom_current)/5),5)))
                for j in range(len(orb_atom_current)):
                    if orb_atom_current[j,2] == 1: #p
                        if orb_atom_current[j,3] == -1:
                            orb_atom_current[j,3] = 0
                        elif orb_atom_current[j,3] == 0:
                            orb_atom_current[j,3] = 1
                        elif orb_atom_current[j,3] == 1:
                            orb_atom_current[j,3] = -1
                    if orb_atom_current[j,2] == 2: #d
                        if orb_atom_current[j,3] == -2:
                            orb_atom_current[j,3] = 0
                        elif orb_atom_current[j,3] == -1:
                            orb_atom_current[j,3] = 2
                        elif orb_atom_current[j,3] == 0:
                            orb_atom_current[j,3] = -2
                        elif orb_atom_current[j,3] == 1:
                            orb_atom_current[j,3] = 1
                        elif orb_atom_current[j,3] == 2:
                            orb_atom_current[j,3] = -1
                    if orb_atom_current[j,2] == 3: #f
                        if orb_atom_current[j,3] == -3:
                            orb_atom_current[j,3] = 0
                        elif orb_atom_current[j,3] == -2:
                            orb_atom_current[j,3] = 1
                        elif orb_atom_current[j,3] == -1:
                            orb_atom_current[j,3] = -1
                        elif orb_atom_current[j,3] == 0:
                            orb_atom_current[j,3] = 2
                        elif orb_atom_current[j,3] == 1:
                            orb_atom_current[j,3] = -2
                        elif orb_atom_current[j,3] == 2:
                            orb_atom_current[j,3] = 3
                        elif orb_atom_current[j,3] == 3:
                            orb_atom_current[j,3] = -3
                sort_index = np.zeros(len(orb_atom_current))
                for j in range(len(orb_atom_current)):
                    sort_index[j] = orb_atom_current[j,3] + 10 * orb_atom_current[j,4] + 100 * orb_atom_current[j,1] + 1000 * orb_atom_current[j,2]
                orb_order = np.argsort(sort_index)
                tmpt = np.empty(len(orb_order))
                for j in range(len(orb_order)):
                    tmpt[orb_order[j]] = j
                orb_order = tmpt
                for j in range(len(orb_atom_current)):
                    orb2deephorb[t,0:3] = np.round(orb_indx[t,12:15])
                    orb2deephorb[t,3] = ia2Riua[int(orb_indx[t,1])-1,3]
                    orb2deephorb[t,4] = int(orb_order[j])
                    t += 1
                atom_current += 1
                orb_atom_current = np.empty((0))

    orb_atom_current = np.reshape(orb_atom_current,((int(len(orb_atom_current)/5),5)))
    for j in range(len(orb_atom_current)):
        if orb_atom_current[j,2] == 1:
            if orb_atom_current[j,3] == -1:
                orb_atom_current[j,3] = 0
            elif orb_atom_current[j,3] == 0:
                orb_atom_current[j,3] = 1
            elif orb_atom_current[j,3] == 1:
                orb_atom_current[j,3] = -1
        if orb_atom_current[j,2] == 2:
            if orb_atom_current[j,3] == -2:
                orb_atom_current[j,3] = 0
            elif orb_atom_current[j,3] == -1:
                orb_atom_current[j,3] = 2
            elif orb_atom_current[j,3] == 0:
                orb_atom_current[j,3] = -2
            elif orb_atom_current[j,3] == 1:
                orb_atom_current[j,3] = 1
            elif orb_atom_current[j,3] == 2:
                orb_atom_current[j,3] = -1
        if orb_atom_current[j,2] == 3: #f
            if orb_atom_current[j,3] == -3:
                orb_atom_current[j,3] = 0
            elif orb_atom_current[j,3] == -2:
                orb_atom_current[j,3] = 1
            elif orb_atom_current[j,3] == -1:
                orb_atom_current[j,3] = -1
            elif orb_atom_current[j,3] == 0:
                orb_atom_current[j,3] = 2
            elif orb_atom_current[j,3] == 1:
                orb_atom_current[j,3] = -2
            elif orb_atom_current[j,3] == 2:
                orb_atom_current[j,3] = 3
            elif orb_atom_current[j,3] == 3:
                orb_atom_current[j,3] = -3
    sort_index = np.zeros(len(orb_atom_current))
    for j in range(len(orb_atom_current)):
        sort_index[j] = orb_atom_current[j,3] + 10 * orb_atom_current[j,4] + 100 * orb_atom_current[j,1] + 1000 * orb_atom_current[j,2]
    orb_order = np.argsort(sort_index)
    tmpt = np.empty(len(orb_order))
    for j in range(len(orb_order)):
        tmpt[orb_order[j]] = j
    orb_order = tmpt
    for j in range(len(orb_atom_current)):
        orb2deephorb[t,0:3] = np.round(orb_indx[t,12:15])
        orb2deephorb[t,3] = ia2Riua[int(orb_indx[t,1])-1,3]
        orb2deephorb[t,4] = int(orb_order[j])
        t += 1

    # Read Useful info of HSX, We only need H and S from this file, but due to structure of fortran unformatted, extra information must be read
    f = FortranFile('{}/{}.HSX'.format(input_path,system_name), 'r')
    tmpt = f.read_ints() # no_u, no_s, nspin, nh
    no_u = tmpt[0]
    no_s = tmpt[1]
    nspin = tmpt[2]
    nh = tmpt[3]
    tmpt = f.read_ints() # gamma
    tmpt = f.read_ints() # indxuo
    tmpt = f.read_ints() # numh
    maxnumh = max(tmpt)
    listh = np.zeros((no_u, maxnumh),dtype=int)
    for i in range(no_u):
        tmpt=f.read_ints() # listh
        for j in range(len(tmpt)):
            listh[i,j] = tmpt[j]

    # finds set of connected atoms
    connected_atoms = set()
    for i in range(no_u):
        for j in range(maxnumh):
            if listh[i,j] == 0:
                #print(j)
                break
            else:
                atom_1 = int(orb2deephorb[i,3])#orbit i belongs to atom_1
                atom_2 = int(orb2deephorb[listh[i,j]-1,3])# orbit j belongs to atom_2
                Rijk = orb2deephorb[listh[i,j]-1,0:3]
                Rijk = Rijk.astype(int)
            connected_atoms = connected_atoms | set(['[{}, {}, {}, {}, {}]'.format(Rijk[0],Rijk[1],Rijk[2],atom_1,atom_2)])


    H_block_sparse = dict()
    for atom_pair in connected_atoms:
        H_block_sparse[atom_pair] = []
    # converts csr-like matrix into coo form in atomic pairs
    for i in range(nspin):
        for j in range(no_u):
            tmpt=f.read_reals(dtype='<f4') # Hamiltonian
            for k in range(len(tmpt)):
                m = 0 # several orbits in siesta differs with DeepH in a (-1) factor
                i2 = j
                j2 = k
                atom_1 = int(orb2deephorb[i2,3])
                m += orb_indx[i2,7]
                atom_2 = int(orb2deephorb[listh[i2,j2]-1,3])
                m += orb_indx[listh[i2,j2]-1,7]
                Rijk = orb2deephorb[listh[i2,j2]-1,0:3]
                Rijk = Rijk.astype(int)
                H_block_sparse['[{}, {}, {}, {}, {}]'.format(Rijk[0],Rijk[1],Rijk[2],atom_1,atom_2)].append([int(orb2deephorb[i2,4]),int(orb2deephorb[listh[i2,j2]-1,4]),tmpt[k]*((-1)**m)])
                pass

    S_block_sparse = dict()
    for atom_pair in connected_atoms:
        S_block_sparse[atom_pair] = []

    for j in range(no_u):
        tmpt=f.read_reals(dtype='<f4') # Overlap
        for k in range(len(tmpt)):
            m = 0
            i2 = j
            j2 = k
            atom_1 = int(orb2deephorb[i2,3])
            m += orb_indx[i2,7]
            atom_2 = int(orb2deephorb[listh[i2,j2]-1,3])
            m += orb_indx[listh[i2,j2]-1,7]
            Rijk = orb2deephorb[listh[i2,j2]-1,0:3]
            Rijk = Rijk.astype(int)
            S_block_sparse['[{}, {}, {}, {}, {}]'.format(Rijk[0],Rijk[1],Rijk[2],atom_1,atom_2)].append([int(orb2deephorb[i2,4]),int(orb2deephorb[listh[i2,j2]-1,4]),tmpt[k]*((-1)**m)])
            pass
        pass

    # finds number of orbitals of each atoms
    nua = int(max(orb2deephorb[:,3]))
    atom2nu = np.zeros(nua)
    for i in range(len(orb_indx)):
        if orb_indx[i,12]==0 and orb_indx[i,13]==0 and orb_indx[i,14]==0:
            if orb_indx[i,4] > atom2nu[int(orb_indx[i,1])-1]:
                atom2nu[int(orb_indx[i,1]-1)] = int(orb_indx[i,4])

    # converts coo sparse matrix into full matrix
    for Rijkab in H_block_sparse.keys():
        sparse_form = H_block_sparse[Rijkab]
        ia1 = int(Rijkab[1:-1].split(',')[3])
        ia2 = int(Rijkab[1:-1].split(',')[4])
        tmpt = np.zeros((int(atom2nu[ia1-1]),int(atom2nu[ia2-1])))
        for i in range(len(sparse_form)):
            tmpt[int(sparse_form[i][0]),int(sparse_form[i][1])]=sparse_form[i][2]/0.036749324533634074/2
        H_block_sparse[Rijkab]=tmpt
    f.close()
    f = h5py.File('{}/hamiltonians.h5'.format(output_path),'w')
    for Rijkab in H_block_sparse.keys():
        f[Rijkab] = H_block_sparse[Rijkab]

    for Rijkab in S_block_sparse.keys():
        sparse_form = S_block_sparse[Rijkab]
        ia1 = int(Rijkab[1:-1].split(',')[3])
        ia2 = int(Rijkab[1:-1].split(',')[4])
        tmpt = np.zeros((int(atom2nu[ia1-1]),int(atom2nu[ia2-1])))
        for i in range(len(sparse_form)):
            tmpt[int(sparse_form[i][0]),int(sparse_form[i][1])]=sparse_form[i][2]
        S_block_sparse[Rijkab]=tmpt

    f.close()
    f = h5py.File('{}/overlaps.h5'.format(output_path),'w')
    for Rijkab in S_block_sparse.keys():
        f[Rijkab] = S_block_sparse[Rijkab]
    f.close()
