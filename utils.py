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

import cProfile
import pstats
import os
def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper