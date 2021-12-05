import click
import yaml

from Dataset import CustomDataset
from train import train


@click.command()
@click.argument('train_dataset_path')
@click.argument('val_dataset_path')
@click.option('--on_gpu', is_flag=True)
def run(train_dataset_path, val_dataset_path, on_gpu) -> None:
    trainset = CustomDataset(train_dataset_path)
    testset = CustomDataset(val_dataset_path, test=True)

    device = 'cuda' if on_gpu else 'cpu'

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    train(config, trainset, testset, device)


if __name__ == '__main__':
    run()
