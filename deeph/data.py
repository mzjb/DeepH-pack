import warnings
import os
import time
import tqdm

from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathos.multiprocessing import ProcessingPool as Pool

from .graph import get_graph


class HData(InMemoryDataset):
    def __init__(self, raw_data_dir: str, graph_dir: str, interface: str, target: str,
                 dataset_name: str, multiprocessing: int, radius, max_num_nbr,
                 num_l, max_element, create_from_DFT, if_lcmp_graph, separate_onsite, new_sp,
                 default_dtype_torch, nums: int = None, transform=None, pre_transform=None, pre_filter=None):
        """
when interface == 'h5',
raw_data_dir
├── 00
│     ├──rh.h5 / rdm.h5
│     ├──rc.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 01
│     ├──rh.h5 / rdm.h5
│     ├──rc.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 02
│     ├──rh.h5 / rdm.h5
│     ├──rc.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── ...
        """
        self.raw_data_dir = raw_data_dir
        assert dataset_name.find('-') == -1, '"-" can not be included in the dataset name'
        if create_from_DFT:
            way_create_graph = 'FromDFT'
        else:
            way_create_graph = f'{radius}r{max_num_nbr}mn'
        if if_lcmp_graph:
            lcmp_str = f'{num_l}l'
        else:
            lcmp_str = 'WithoutLCMP'
        if separate_onsite is True:
            onsite_str = '-SeparateOnsite'
        else:
            onsite_str = ''
        if new_sp:
            new_sp_str = '-NewSP'
        else:
            new_sp_str = ''
        if target == 'hamiltonian':
            title = 'HGraph'
        else:
            raise ValueError('Unknown prediction target: {}'.format(target))
        graph_file_name = f'{title}-{interface}-{dataset_name}-{lcmp_str}-{way_create_graph}{onsite_str}{new_sp_str}.pkl'
        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)
        self.data, self.slices = None, None
        self.interface = interface
        self.target = target
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.num_l = num_l
        self.create_from_DFT = create_from_DFT
        self.if_lcmp_graph = if_lcmp_graph
        self.separate_onsite = separate_onsite
        self.new_sp = new_sp
        self.default_dtype_torch = default_dtype_torch

        self.nums = nums
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        print(f'Graph data file: {graph_file_name}')
        if os.path.exists(self.data_file):
            print('Use existing graph data file')
        else:
            print('Process new data file......')
            self.process()
        begin = time.time()
        try:
            loaded_data = torch.load(self.data_file)
        except AttributeError:
            raise RuntimeError('Error in loading graph data file, try to delete it and generate the graph file with the current version of PyG')
        if len(loaded_data) == 2:
            warnings.warn('You are using the graph data file with an old version')
            self.data, self.slices = loaded_data
            self.info = {
                "spinful": False,
                "index_to_Z": torch.arange(max_element + 1),
                "Z_to_index": torch.arange(max_element + 1),
            }
        elif len(loaded_data) == 3:
            self.data, self.slices, tmp = loaded_data
            if isinstance(tmp, dict):
                self.info = tmp
                print(f"Atomic types: {self.info['index_to_Z'].tolist()}")
            else:
                warnings.warn('You are using an old version of the graph data file')
                self.info = {
                    "spinful": tmp,
                    "index_to_Z": torch.arange(max_element + 1),
                    "Z_to_index": torch.arange(max_element + 1),
                }
        print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
              f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.0f} seconds')

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]

        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')).T,
                              np.loadtxt(os.path.join(folder, 'element.dat')),
                              np.loadtxt(os.path.join(folder, 'site_positions.dat')).T,
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch)
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch)
        if self.target == 'E_ij':
            huge_structure = True
        else:
            huge_structure = False
        return get_graph(cart_coords, frac_coords, numbers, stru_id, r=self.radius, max_num_nbr=self.max_num_nbr,
                         numerical_tol=1e-8, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         tb_folder=folder, interface=self.interface, num_l=self.num_l,
                         create_from_DFT=self.create_from_DFT, if_lcmp_graph=self.if_lcmp_graph,
                         separate_onsite=self.separate_onsite,
                         target=self.target, huge_structure=huge_structure, if_new_sp=self.new_sp, **kwargs)

    def process(self):
        begin = time.time()
        folder_list = []
        for root, dirs, files in os.walk(self.raw_data_dir):
            if (self.interface == 'h5' and 'rc.h5' in files) or (
                    self.interface == 'npz' and 'rc.npz' in files):
                folder_list.append(root)
        folder_list = sorted(folder_list)
        folder_list = folder_list[: self.nums]
        if self.dataset_name == 'graphene_450':
            folder_list = folder_list[500:5000:10]
        if self.dataset_name == 'graphene_1500':
            folder_list = folder_list[500:5000:3]
        if self.dataset_name == 'bp_bilayer':
            folder_list = folder_list[:600]
        assert len(folder_list) != 0, "Can not find any structure"
        print('Found %d structures, have cost %d seconds' % (len(folder_list), time.time() - begin))

        if self.multiprocessing == 0:
            print(f'Use multiprocessing (nodes = num_processors x num_threads = 1 x {torch.get_num_threads()})')
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list)]
        else:
            pool_dict = {} if self.multiprocessing < 0 else {'nodes': self.multiprocessing}
            # BS (2023.06.06): 
            # The keyword "num_threads" in kernel.py can be used to set the torch threads.
            # The multiprocessing in the "process_worker" is in contradiction with the num_threads utilized in torch.
            # To avoid this conflict, I limit the number of torch threads to one,
            # and recover it when finishing the process_worker.
            torch_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)

            with Pool(**pool_dict) as pool:
                nodes = pool.nodes
                print(f'Use multiprocessing (nodes = num_processors x num_threads = {nodes} x {torch.get_num_threads()})')
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list)))
            torch.set_num_threads(torch_num_threads)
        print('Finish processing %d structures, have cost %d seconds' % (len(data_list), time.time() - begin))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        index_to_Z, Z_to_index = self.element_statistics(data_list)
        spinful = data_list[0].spinful
        for d in data_list:
            assert spinful == d.spinful

        data, slices = self.collate(data_list)
        torch.save((data, slices, dict(spinful=spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index)), self.data_file)
        print('Finish saving %d structures to %s, have cost %d seconds' % (
        len(data_list), self.data_file, time.time() - begin))

    def element_statistics(self, data_list):
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))

        for data in data_list:
            data.x = Z_to_index[data.x]

        return index_to_Z, Z_to_index
