import time
import torch

def get_time():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))

def log(*args):
    print('%s %s' % (get_time(), ' '.join([str(it) for it in args])), flush=True)

def save_model(net, filename):
    if hasattr(net, 'module'):
        torch.save(net.module.state_dict(), filename)
    else:
        torch.save(net.state_dict(), filename)