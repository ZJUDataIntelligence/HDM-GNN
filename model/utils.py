import pickle
import numpy as np
import torch
import random

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mae_np(output, target):
    return np.mean(np.absolute(output - target))

def mse_np(output, target):
    return ((output - target) ** 2).mean()

def rmse_np(output, target):
    return np.sqrt(((output - target) ** 2).mean())

def mape_np(output, target):
    output = output.flatten()
    target = target.flatten()
    b = np.stack((output, target), axis=0)
    mask = (b[1] == 0)
    b = b[:, ~mask]
    output_mask = b[0]
    target_mask = b[1]
    return np.mean(np.abs((output_mask - target_mask) / target_mask)) * 100

# if __name__ == '__main__':
#     c = [1, 2, 3, 4, 5, 6, 7]
#     filename = save_variable(c, '/home/zh/temp_v')
#     d = load_variavle(filename)
#     print(d == c)