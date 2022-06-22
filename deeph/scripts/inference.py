import os
import time
import subprocess as sp
import json

import argparse

from deeph import get_inference_config, rotate_back
from deeph.preprocess import openmx_parse_overlap, get_rc
from deeph.inference import predict, predict_with_grad


def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_inference_config(args.config)

    work_dir = os.path.abspath(config.get('basic', 'work_dir'))
    OLP_dir = os.path.abspath(config.get('basic', 'OLP_dir'))
    structure_file_name = config.get('basic', 'structure_file_name')
    task = json.loads(config.get('basic', 'task'))
    assert isinstance(task, list)
    disable_cuda = config.getboolean('basic', 'disable_cuda')
    device = config.get('basic', 'device')
    huge_structure = config.getboolean('basic', 'huge_structure')
    restore_blocks_py = config.getboolean('basic', 'restore_blocks_py')
    gen_rc_idx = config.getboolean('basic', 'gen_rc_idx')
    gen_rc_by_idx = config.get('basic', 'gen_rc_by_idx')
    with_grad = config.getboolean('basic', 'with_grad')
    julia_interpreter = config.get('interpreter', 'julia_interpreter')
    radius = config.getfloat('graph', 'radius')

    if with_grad:
        assert restore_blocks_py is True
        assert 4 not in task
        assert 5 not in task

    os.makedirs(work_dir, exist_ok=True)
    config.write(open(os.path.join(work_dir, 'config.ini'), "w"))

    if not restore_blocks_py:
        cmd3_post = f"{julia_interpreter} " \
                    f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'restore_blocks.jl')} " \
                    f"--input_dir {work_dir} --output_dir {work_dir}"
    cmd5 = f"{julia_interpreter} " \
           f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'sparse_calc.jl')} " \
           f"--input_dir {work_dir} --output_dir {work_dir} --config {config.get('basic', 'sparse_calc_config')}"

    print(f"\n~~~~~~~ 1.parse_Overlap\n")
    print(f"\n~~~~~~~ 2.get_local_coordinate\n")
    print(f"\n~~~~~~~ 3.get_pred_Hamiltonian\n")
    if not restore_blocks_py:
        print(f"\n~~~~~~~ 3_post.restore_blocks, command: \n{cmd3_post}\n")
    print(f"\n~~~~~~~ 4.rotate_back\n")
    print(f"\n~~~~~~~ 5.sparse_calc, command: \n{cmd5}\n")

    if 1 in task:
        begin = time.time()
        print(f"\n####### Begin 1.parse_Overlap")
        openmx_parse_overlap(OLP_dir, work_dir, os.path.join(OLP_dir, structure_file_name))
        assert os.path.exists(os.path.join(work_dir, "overlaps.h5"))
        assert os.path.exists(os.path.join(work_dir, "lat.dat"))
        assert os.path.exists(os.path.join(work_dir, "rlat.dat"))
        assert os.path.exists(os.path.join(work_dir, "site_positions.dat"))
        assert os.path.exists(os.path.join(work_dir, "orbital_types.dat"))
        assert os.path.exists(os.path.join(work_dir, "element.dat"))
        assert os.path.exists(os.path.join(work_dir, "R_list.dat"))
        print('\n******* Finish 1.parse_Overlap, cost %d seconds\n' % (time.time() - begin))

    if not with_grad and 2 in task:
        begin = time.time()
        print(f"\n####### Begin 2.get_local_coordinate")
        get_rc(work_dir, work_dir, radius=radius, gen_rc_idx=gen_rc_idx, gen_rc_by_idx=gen_rc_by_idx,
               create_from_DFT=config.getboolean('graph', 'create_from_DFT'))
        assert os.path.exists(os.path.join(work_dir, "rc.h5"))
        print('\n******* Finish 2.get_local_coordinate, cost %d seconds\n' % (time.time() - begin))

    if 3 in task:
        begin = time.time()
        print(f"\n####### Begin 3.get_pred_Hamiltonian")
        trained_model_dir = config.get('basic', 'trained_model_dir')
        if trained_model_dir[0] == '[' and trained_model_dir[-1] == ']':
            trained_model_dir = json.loads(trained_model_dir)
        if with_grad:
            predict_with_grad(input_dir=work_dir, output_dir=work_dir, disable_cuda=disable_cuda, device=device,
                              huge_structure=huge_structure, trained_model_dirs=trained_model_dir)
        else:
            predict(input_dir=work_dir, output_dir=work_dir, disable_cuda=disable_cuda, device=device,
                    huge_structure=huge_structure, restore_blocks_py=restore_blocks_py,
                    trained_model_dirs=trained_model_dir)
        if restore_blocks_py:
            if with_grad:
                assert os.path.exists(os.path.join(work_dir, "hamiltonians_grad_pred.h5"))
                assert os.path.exists(os.path.join(work_dir, "hamiltonians_pred.h5"))
            else:
                assert os.path.exists(os.path.join(work_dir, "rh_pred.h5"))
        else:
            capture_output = sp.run(cmd3_post, shell=True, capture_output=False, encoding="utf-8")
            assert capture_output.returncode == 0
            assert os.path.exists(os.path.join(work_dir, "rh_pred.h5"))
        print('\n******* Finish 3.get_pred_Hamiltonian, cost %d seconds\n' % (time.time() - begin))

    if 4 in task:
        begin = time.time()
        print(f"\n####### Begin 4.rotate_back")
        rotate_back(input_dir=work_dir, output_dir=work_dir)
        assert os.path.exists(os.path.join(work_dir, "hamiltonians_pred.h5"))
        print('\n******* Finish 4.rotate_back, cost %d seconds\n' % (time.time() - begin))

    if 5 in task:
        begin = time.time()
        print(f"\n####### Begin 5.sparse_calc")
        capture_output = sp.run(cmd5, shell=True, capture_output=False, encoding="utf-8")
        assert capture_output.returncode == 0
        assert os.path.exists(os.path.join(work_dir, "sparse_matrix.jld"))
        print('\n******* Finish 5.sparse_calc, cost %d seconds\n' % (time.time() - begin))


if __name__ == '__main__':
    main()
