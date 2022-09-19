import os
import subprocess as sp
import time

import argparse
from pathos.multiprocessing import ProcessingPool as Pool

from deeph import get_preprocess_config, get_rc, get_rh, abacus_parse, siesta_parse


def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_preprocess_config(args.config)

    raw_dir = os.path.abspath(config.get('basic', 'raw_dir'))
    processed_dir = os.path.abspath(config.get('basic', 'processed_dir'))
    abacus_suffix = str(config.get('basic', 'abacus_suffix', fallback='ABACUS'))
    target = config.get('basic', 'target')
    interface = config.get('basic', 'interface')
    local_coordinate = config.getboolean('basic', 'local_coordinate')
    multiprocessing = config.getint('basic', 'multiprocessing')
    get_S = config.getboolean('basic', 'get_S')

    julia_interpreter = config.get('interpreter', 'julia_interpreter')

    def make_cmd(input_dir, output_dir, target, interface, get_S):
        if interface == 'openmx':
            if target == 'hamiltonian':
                cmd = f"{julia_interpreter} " \
                      f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess', 'openmx_get_data.jl')} " \
                      f"--input_dir {input_dir} --output_dir {output_dir} --save_overlap {str(get_S).lower()}"
            elif target == 'density_matrix':
                cmd = f"{julia_interpreter} " \
                      f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess', 'openmx_get_data.jl')} " \
                      f"--input_dir {input_dir} --output_dir {output_dir} --save_overlap {str(get_S).lower()} --if_DM true"
            else:
                raise ValueError('Unknown target: {}'.format(target))
        elif interface == 'siesta' or interface == 'abacus':
            cmd = ''
        elif interface == 'aims':
            cmd = f"{julia_interpreter} " \
                  f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess', 'aims_get_data.jl')} " \
                  f"--input_dir {input_dir} --output_dir {output_dir} --save_overlap {str(get_S).lower()}"
        else:
            raise ValueError('Unknown interface: {}'.format(interface))
        return cmd

    os.chdir(raw_dir)
    relpath_list = []
    abspath_list = []
    for root, dirs, files in os.walk('./'):
        if (interface == 'openmx' and 'openmx.scfout' in files) or (
            interface == 'abacus' and 'OUT.' + abacus_suffix in dirs) or (
            interface == 'siesta' and any(['.HSX' in ifile for ifile in files])) or (
            interface == 'aims' and 'NoTB.dat' in files):
            relpath_list.append(root)
            abspath_list.append(os.path.abspath(root))

    os.makedirs(processed_dir, exist_ok=True)
    os.chdir(processed_dir)
    print(f"Found {len(abspath_list)} directories to preprocess")

    def worker(index):
        time_cost = time.time() - begin_time
        current_block = index // nodes
        if current_block < 1:
            time_estimate = '?'
        else:
            num_blocks = (len(abspath_list) + nodes - 1) // nodes
            time_estimate = time.localtime(time_cost / (current_block) * (num_blocks - current_block))
            time_estimate = time.strftime("%H:%M:%S", time_estimate)
        print(f'\rPreprocessing No. {index + 1}/{len(abspath_list)} '
              f'[{time.strftime("%H:%M:%S", time.localtime(time_cost))}<{time_estimate}]...', end='')
        abspath = abspath_list[index]
        relpath = relpath_list[index]
        os.makedirs(relpath, exist_ok=True)
        cmd = make_cmd(
            abspath,
            os.path.abspath(relpath),
            target=target,
            interface=interface,
            get_S=get_S,
        )
        capture_output = sp.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if capture_output.returncode != 0:
            with open(os.path.join(os.path.abspath(relpath), 'error.log'), 'w') as f:
                f.write(f'[stdout of cmd "{cmd}"]:\n\n{capture_output.stdout}\n\n\n'
                        f'[stderr of cmd "{cmd}"]:\n\n{capture_output.stderr}')
            print(f'\nFailed to preprocess: {abspath}, '
                  f'log file was saved to {os.path.join(os.path.abspath(relpath), "error.log")}')
            return

        if interface == 'abacus':
            print("Output subdirectories:", "OUT." + abacus_suffix)
            abacus_parse(abspath, os.path.abspath(relpath), 'OUT.' + abacus_suffix)
        elif interface == 'siesta':
            siesta_parse(abspath, os.path.abspath(relpath))
        if local_coordinate:
            get_rc(os.path.abspath(relpath), os.path.abspath(relpath), radius=config.getfloat('graph', 'radius'),
                   r2_rand=config.getboolean('graph', 'r2_rand'),
                   create_from_DFT=config.getboolean('graph', 'create_from_DFT'), neighbour_file='hamiltonians.h5')
            get_rh(os.path.abspath(relpath), os.path.abspath(relpath), target)

    begin_time = time.time()
    if multiprocessing != 0:
        if multiprocessing > 0:
            pool_dict = {'nodes': multiprocessing}
        else:
            pool_dict = {}
        with Pool(**pool_dict) as pool:
            nodes = pool.nodes
            print(f'Use multiprocessing (nodes = {nodes})')
            pool.map(worker, range(len(abspath_list)))
    else:
        nodes = 1
        for index in range(len(abspath_list)):
            worker(index)
    print(f'\nPreprocess finished in {time.time() - begin_time:.2f} seconds')

if __name__ == '__main__':
    main()
