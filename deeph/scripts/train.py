import argparse

from deeph import DeepHKernel, get_config


def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_config(args.config)
    only_get_graph = config.getboolean('basic', 'only_get_graph')
    kernel = DeepHKernel(config)
    train_loader, val_loader, test_loader, transform = kernel.get_dataset(only_get_graph)
    if only_get_graph:
        return
    kernel.build_model()
    kernel.set_train()
    kernel.train(train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()
