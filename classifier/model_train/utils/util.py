
import random
import sys
import os
import torch
import numpy as np
from torch.nn import init
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
import datetime
import platform
from models import *
from copy import deepcopy

def select_loss_function(loss_function):
    if loss_function == 'L1LOSS':
        criterion = nn.L1Loss()
    elif loss_function == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

def select_optim(config, model_prams):
    if config['optim'] == 'ADAM':
        optimizer = optim.Adam(model_prams, lr=config['lr_init'], betas=(config['beta_1'], config['beta_2']))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(model_prams, lr=config['lr_init'], momentum = config['momentum'], weight_decay=5e-4)
    return optimizer


#  class LambdaLR_():
#      def __init__(self, n_epochs, offset, decay_start_epoch):
#          assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
#          self.n_epochs = n_epochs
#          self.offset = offset
#          self.decay_start_epoch = decay_start_epoch

#      def step(self, epoch):
#          return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def select_lr_scheduler(opt, optimizer):
    if opt.lr_scheduler == "STEPLR":
        lr_scheduler = StepLR(optimizer, step_size=opt.decay_epoch, gamma=opt.decay_rate)
    elif opt.lr_scheduler == "LAMBDALR":
        lr_scheduler = LambdaLR(optimizer, lr_lambda= lambda epoch: 0.95 ** epoch)
    elif opt.lr_scheduler == "EXPLR":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.1)
    else:
        lr_scheduler = None
    return lr_scheduler


class LearningRateWarmUP(object):
    def __init__(self, warmup_epoch, target_lr, optimizer, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_epoch
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            if self.after_scheduler != None:
                self.after_scheduler.step()


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]

def fitness(x):
    # acc, recall, precision, speciticity, f_1
    # Model fitness as a weighted combination of metrics
    w = [0.8, 0.05, 0.05, 0.05, 0.05]  # weights for [Acc, Precision, Recall, Specificity, F1 score]
    return (x * w).sum()



class EarlyStopping:
    def __init__(self, patience, delta, early_criterion, verbose=False, loss_score=None, acc_score=None):
        self.patience = patience
        self.delta = delta
        self.early_criterion = early_criterion

        self.verbose = verbose
        self.counter = 0
        self.best_loss_score = loss_score
        self.best_acc_score = acc_score
        self.early_stop = False


    def __call__(self, acc, val_loss):
        if self.early_criterion=='ACCURACY':
            acc_score = acc
            print(acc_score)
            if self.best_acc_score is None:
                self.best_acc_score = acc_score
            elif acc_score < self.best_acc_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_acc_score = acc_score
                self.counter = 0
        else:
            loss_score = val_loss
            print(loss_score)
            if self.best_loss_score is None:
                self.best_loss_score = loss_score
            elif loss_score > self.best_loss_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss_score = loss_score
                self.counter = 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count




def select_device(logger, device='cpu', batch_size=None):
    s = f'ResNet 🚀 {date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        space = ' ' * len(s)
        p = torch.cuda.get_device_properties(int(device))
        s += f"{space}CUDA:{device} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
        device = torch.device('cuda:'+device)

    else:
        s += 'CPU\n'
        device = torch.device('cpu')
    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return device
def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'



def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)



# import random
# import sys
# import os
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# import torch
# import numpy as np
# from torch.nn import init


# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# def init_weight(net, init_type='normal', init_gain=0.02):
    # def init_func(m):
        # classname = m.__class__.__name__
        # if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # if init_type == 'normal':
                # init.normal_(m.weight.data, 0.0, init_gain)
            # elif init_type == 'xavier':
                # init.xavier_normal_(m.weight.data, gain=init_gain)
            # elif init_type == 'kaiming':
                # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            # elif init_type == 'orthogonal':
                # init.orthogonal_(m.weight.data, gain=init_gain)

            # if hasattr(m, 'bias') and m.bias is not None:
                # init.constant_(m.bias.data, 0.0)

        # elif classname.find('BatchNorm2d') != -1:
            # init.normal_(m.weight.data, 1.0, init_gain)
            # init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    # net.apply(init_func)

# def save(ckpt_path, net, optim, epoch):
    # if os.path.exists(ckpt_path) == False:
        # os.makedirs(ckpt_path)
    # torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},'%s/net_epoch%d.pth' %(ckpt_path, epoch))

# def load(ckpt_path, net, optim):
    # if os.path.exists(ckpt_path) == False:
        # os.mkdir(ckpt_path)
    # ckpt_list = os.listdir(ckpt_path)
    # ckpt_list.sort()
    # model_dict = torch.load('%s/%s' % ckpt_path, ckpt_list[-1])
    # net.load_state_dict(model_dict['net'])
    # optim.load_state_dict(model_dict['optim'])
    # return net, optim

# def matplotlib_imshow(img, one_channel=False):
    # if one_channel:
        # img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.to('cpu').numpy()
    # if one_channel:
        # plt.imshow(npimg, cmap="Greys")
    # else:
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))

# def images_to_probs(net, images):
    # output = net(images)
    # # convert output probabilities to predicted class
    # _, preds_tensor = torch.max(output, 1)
    # preds = np.squeeze(preds_tensor.to('cpu').numpy())
    # return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

# def plot_classes_preds(net, images, labels):
    # preds, probs = images_to_probs(net, images)
    # fig = plt.figure(figsize=(12, 48))
    # for idx in np.arange(4):
        # ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        # matplotlib_imshow(images[idx], one_channel=True)
        # ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            # classes[preds[idx]],
            # probs[idx] * 100.0,
            # classes[labels[idx]]),
                    # color=("green" if preds[idx]==labels[idx].item() else "red"))
    # return fig

