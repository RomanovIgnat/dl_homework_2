import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dataset import CustomDataset
from torchvision.models import vgg16_bn


@click.command()
@click.argument('train_dataset_path')
@click.argument('val_dataset_path')
@click.option('--on_gpu', is_flag=True)
def run(train_dataset_path, val_dataset_path, on_gpu) -> None:
    trainset = CustomDataset(train_dataset_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    testset = CustomDataset(val_dataset_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=True, num_workers=0)
    test_size = len(testset)

    device = 'cuda' if on_gpu else 'cpu'

    model = vgg16_bn(pretrained=False, progress=True, num_classes=200)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    num_of_epochs = 100

    for epoch in range(num_of_epochs):

        if epoch and not epoch % 8:
            opt.param_groups[0]['lr'] /= 2

        model.train(True)
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)

            if not i % 100:
                print(loss.to('cpu').data.numpy(), end=" ", flush=True)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.train(False)
        with torch.no_grad():
            res = 0
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                _, labels = torch.max(preds, 1)
                res += (labels == y).sum().to('cpu').data.numpy()

            print("acc: ", res / test_size)


if __name__ == '__main__':
    run()
