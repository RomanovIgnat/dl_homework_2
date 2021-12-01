import click
import torch

from Dataset import CustomDataset


@click.command()
@click.argument('dataset_path')
def run(dataset_path) -> None:
    dataset = CustomDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    for i, (x, y) in enumerate(dataloader):
        if not i % 10:
            print(i % 10)
        continue


if __name__ == '__main__':
    run()
