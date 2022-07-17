import os
import subprocess as sp

import argparse
from pathos.multiprocessing import ProcessingPool as Pool

from deeph import get_preprocess_config, get_rc, get_rh


def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_preprocess_config(args.config)

    raw_dir = os.path.abspath(config.get('basic', 'raw_dir'))
    processed_dir = os.path.abspath(config.get('basic', 'processed_dir'))
    target = config.get('basic', 'target')
    interface = config.get('basic', 'interface')

    julia_interpreter = config.get('interpreter', 'julia_interpreter')

    def make_cmd(input_dir, output_dir, target, interface):
        if interface == 'openmx':
            if target == 'hamiltonian':
                cmd = f"{julia_interpreter} " \
                      f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess', 'openmx_get_data.jl')} " \
                      f"--input_dir {input_dir} --output_dir {output_dir}"
            elif target == 'density_matrix':
                cmd = f"{julia_interpreter} " \
                      f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocess', 'openmx_get_data.jl')} " \
                      f"--input_dir {input_dir} --output_dir {output_dir} --if_DM true"
            else:
                raise ValueError('Unknown target: {}'.format(target))
        elif interface == 'siesta':
            cmd = ''
        else:
            raise ValueError('Unknown interface: {}'.format(interface))
        return cmd

    os.chdir(raw_dir)
    relpath_list = []
    abspath_list = []
    for root, dirs, files in os.walk('./'):
        if (interface == 'openmx' and 'openmx.scfout' in files) or (
                interface == 'siesta' and 'hamiltonians.h5' in files):
            relpath_list.append(root)
            abspath_list.append(os.path.abspath(root))

    os.makedirs(processed_dir, exist_ok=True)
    os.chdir(processed_dir)
    print(f"Found {len(abspath_list)} directories to preprocess")

    def worker(index):
        print(f'Preprocessing No. {index + 1}/{len(abspath_list)}...')
        abspath = abspath_list[index]
        relpath = relpath_list[index]
        os.makedirs(relpath, exist_ok=True)
        cmd = make_cmd(
            abspath,
            os.path.abspath(relpath),
            target=target,
            interface=interface
        )
        capture_output = sp.run(cmd, shell=True, capture_output=False, encoding="utf-8")
        assert capture_output.returncode == 0
        get_rc(os.path.abspath(relpath), os.path.abspath(relpath), radius=config.getfloat('graph', 'radius'),
               r2_rand=config.getboolean('graph', 'r2_rand'),
               create_from_DFT=config.getboolean('graph', 'create_from_DFT'), neighbour_file='hamiltonians.h5')
        get_rh(os.path.abspath(relpath), os.path.abspath(relpath), target)

    if config.getboolean('basic', 'multiprocessing'):
        print('Use multiprocessing')
        with Pool() as pool:
            pool.imap(worker, range(len(abspath_list)))
    else:
        for index in range(len(abspath_list)):
            worker(index)

if __name__ == '__main__':
    main()
