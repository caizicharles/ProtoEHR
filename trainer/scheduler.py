import torch.optim as optim

SCHEDULERS = {'StepLR': optim.lr_scheduler.StepLR,
              'ExpLR': optim.lr_scheduler.ExponentialLR}
