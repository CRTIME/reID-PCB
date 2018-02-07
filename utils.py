import time

def get_time():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))

def log(*args):
    print('%s %s' % (get_time(), ' '.join([str(it) for it in args])))