'''
Author: kavinbj
Date: 2023-02-04 14:02:05
LastEditTime: 2023-02-04 21:07:35
FilePath: qutils.py
Description: 

Copyright (c) 2023 by ${git_name}, All Rights Reserved. 
'''

from __future__ import annotations
import os
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data
import time
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

from torch.ao.quantization import disable_observer
from torch.ao.quantization.quantize import convert, prepare_qat
from torch.ao.quantization.qconfig import get_default_qat_qconfig

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches=-1):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            print('idx', idx)
            output = model(image)
            _ = criterion(output, target)
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            #print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if neval_batches == -1:
                continue
            else:
                cnt += 1
                if cnt >= neval_batches:
                    return top1, top5
    return top1, top5


def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def get_size_of_model(model_path):
    '''获取模型的大小（MB）'''
    return os.path.getsize(model_path)/1e6

def print_size_of_model(model, model_path="temp.p"):
    torch.save(model.state_dict(), model_path)
    print(f'模型大小：{get_size_of_model(model_path)} MB', )
    os.remove('temp.p')


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches=-1):
    model = model.to(device)
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        # start_time = time.time()
        print('.', end='')
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if ntrain_batches == -1:
            continue
        else:
            cnt += 1
            if cnt >= ntrain_batches:
                print('Loss', avgloss.avg)
                print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))
                return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
        """Download the Fashion-MNIST dataset and then load it into memory.
        Defined in :numref:`sec_fashion_mnist`"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../datasets", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../datasets", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))


def load_data_cifar10(batch_size, resize=None, num_workers=4):
        """Download the Cifar10 dataset and then load it into memory."""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        _train = torchvision.datasets.CIFAR10(
            root="../datasets", train=True, transform=trans, download=True)
        _test = torchvision.datasets.CIFAR10(
            root="../datasets", train=False, transform=trans, download=True)
        return (data.DataLoader(_train, batch_size, shuffle=True,
                                num_workers=num_workers),
                data.DataLoader(_test, batch_size, shuffle=False,
                                num_workers=num_workers))

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.
    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot 多张图片
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

class Fx:
    ones = torch.ones
    zeros = torch.zeros
    tensor = torch.tensor
    arange = torch.arange
    meshgrid = torch.meshgrid
    sin = torch.sin
    sinh = torch.sinh
    cos = torch.cos
    cosh = torch.cosh
    tanh = torch.tanh
    linspace = torch.linspace
    exp = torch.exp
    log = torch.log
    normal = torch.normal
    rand = torch.rand
    matmul = torch.matmul
    int32 = torch.int32
    float32 = torch.float32
    concat = torch.cat
    stack = torch.stack
    abs = torch.abs
    eye = torch.eye
    numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
    size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
    reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
    to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)


def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = Fx.argmax(y_hat, axis=1)
    cmp = Fx.astype(y_hat, y.dtype) == y
    return float(Fx.reduce_sum(Fx.astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter, device='cpu'):
    """计算在指定数据集上模型的精度
    """
    net = net.to(device)
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), Fx.size(y))
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), Fx.size(y))
    return metric[0] / metric[1]


def train_batch(net, X, y, loss, trainer, device):
    """Train for a minibatch with mutiple GPUs.
    """
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter,
          loss, trainer, num_epochs,
          device='cpu',
          need_qconfig=False,
          is_freeze=False,
          is_quantized_acc=False,
          backend='fbgemm',
          ylim=[0, 1]):
    """Train a model with mutiple GPUs.
    """
    timer, num_batches = Timer(), len(train_iter)
    _ylim = '' if ylim[0] == 0 else f'{ylim[0]}+'
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=ylim,
                        legend=[f'{_ylim}train loss', 'train acc', 'test acc'])
    # nn.DataParallel(net, device_ids=devices).to(devices[0])
    net = net.to(device)
    # net = net.to('cuda')
    if device != 'cpu':
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
 
    if need_qconfig:
        # net.fuse_model()
        net.qconfig = get_default_qat_qconfig(backend)
        # net = prepare_qat(net)
    for epoch in range(num_epochs):
        print('epoch', epoch)
        metric = Accumulator(4)
        if is_freeze:
            if epoch > 3:
                # 冻结 quantizer 参数
                net.apply(disable_observer)
            if epoch > 2:
                # 冻结 batch 的平均值和方差估计
                net.apply(nn.intrinsic.qat.freeze_bn_stats)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features,
                                    labels, loss,
                                    trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # print((metric[0] / metric[2])+ylim[0])
                animator.add(epoch + (i + 1) / num_batches,
                             ((metric[0] / metric[2])+ylim[0], metric[1] / metric[3],
                             None))
        if is_quantized_acc:
            quantized_model = deepcopy(net).to('cpu').eval()
            quantized_model = convert(quantized_model, inplace=False)
            test_acc = evaluate_accuracy(quantized_model, test_iter)
        else:
            test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(device)}')

def train_fine_tuning(net,
                      train_iter, test_iter,
                      learning_rate,
                      num_epochs=5,
                      device='cuda:0',
                      is_freeze=False,
                      is_quantized_acc=False,
                      need_qconfig=False,
                      param_group=True,
                      ylim=[0, 1],
                      output_layer='classifier'):
    # 如果param_group=True，输出层中的模型参数将使用十倍的学习率
    # param_name 可能为 'fc' 或者 'classifier'
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name.split('.')[0] != output_layer]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': getattr(net, output_layer).parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    net.train()
    train(net, train_iter, test_iter, 
          loss, trainer, num_epochs,
          device, ylim=ylim,
          need_qconfig=need_qconfig,
          is_freeze=is_freeze,
          is_quantized_acc=is_quantized_acc)



