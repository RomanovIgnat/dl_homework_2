from model import ClassifierModel
import torch
import torch.nn.functional as F


def train_step(model, optimizer, loss_function, x, y, device, iteration):
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = loss_function(y_pred, y)

    if not iteration % 100:
        print(loss.to('cpu').data.numpy(), end=" ", flush=True)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def validation_step(model, x, y, device):
    x, y = x.to(device), y.to(device)
    preds = model(x)
    _, labels = torch.max(preds, 1)
    return (labels == y).sum().to('cpu').data.numpy()


def train(config, trainset, testset, device):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['train_batchsize'], shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['test_batchsize'], shuffle=False, num_workers=0)
    test_size = len(testset)

    model = ClassifierModel().to(device)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(config['num_of_epochs']):

        if epoch and not epoch % config['change_lr_step']:
            opt.param_groups[0]['lr'] /= 2

        model.train(True)
        for i, (x, y) in enumerate(trainloader):
            train_step(model, opt, F.cross_entropy, x, y, device, i)

        model.train(False)
        with torch.no_grad():
            res = 0
            for x, y in testloader:
                res += validation_step(model, x, y, device)

            print("acc:", res / test_size)


