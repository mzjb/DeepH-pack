import argparse

from deeph import DeepHKernal, get_config


def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_config(args.config)
    kernal = DeepHKernal(config)
    train_loader, val_loader, test_loader, transform = kernal.get_dataset()
    kernal.build_model()
    kernal.set_train()
    kernal.train(train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()
