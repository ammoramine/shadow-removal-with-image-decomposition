import time
import torch
import numpy as np

def timeit(func,*args,**kwargs):
    start = time.time()
    res = func(*args,**kwargs)
    stop = time.time()
    print(stop-start)
    return res


def pils_to_tensor(batch_pils):
    return torch.Tensor(np.array([np.array(el) for el in batch_pils]))
