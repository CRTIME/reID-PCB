import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from PIL import Image

from utils import log
from net import FeatureExtractor
from data import Market1501
from config import transform
from utils import get_time

def extract_feat(args, extractor, dataloader, feat_dim):
    feat = []
    labels = []
    cameras = []
    filenames = []
    for _, data in enumerate(dataloader):
        extractor.eval()
        inputs, l, c, f = data
        inputs = Variable(inputs, volatile=True)
        if args.use_gpu:
            inputs = inputs.cuda()
        outputs = extractor.forward(inputs)
        feat.append(outputs)
        labels += list(l)
        cameras += list(c)
        filenames += list(f)
    feat = torch.cat(feat)
    feat.view(-1, feat_dim)
    return (feat.cpu().data.numpy(), np.array(labels),
           np.array(cameras), np.array(filenames))

def get_dist(query, test):
    return cdist(query, test)

def calc_dist(query_feat, test_feat):
    class Dist(nn.Module):
        def __init__(self):
            super(Dist, self).__init__()
        def forward(self, x, y):
            n = x.size(0)
            m = y.size(0)
            d = x.size(1)
            x = x.unsqueeze(1).expand(n, m, d)
            y = y.unsqueeze(0).expand(n, m, d)
            dist = torch.pow(x - y, 2).sum(2)
            return dist
    pdist = Dist().cuda()
    split_num = 40
    lx = int(len(query_feat) / split_num) + 1
    ly = int(len(test_feat) / split_num) + 1
    dist = []
    for i in range(split_num):
        tmp_dist = []
        if i * lx >= len(query_feat):
            continue
        x = torch.from_numpy(query_feat[i*lx:(i+1)*lx])
        x = Variable(x, volatile=True).cuda()
        for j in range(split_num):
            if j * ly >= len(test_feat):
                continue
            y = torch.from_numpy(test_feat[j*ly:(j+1)*ly])
            y = Variable(y, volatile=True).cuda()
            d = pdist(x, y).cpu().data.numpy()
            tmp_dist.append(d)
        tmp_dist = np.concatenate(tmp_dist, axis=1)
        dist.append(tmp_dist)
    dist = np.concatenate(dist, axis=0)
    return dist

def get_rank_x(x, dist, query_labels, query_cameras, test_labels, test_cameras):
    rank_x = 0
    total = 0
    for i, row in enumerate(dist):
        index = np.argsort(row)
        good = False
        vaild_num = 0
        for j in index:
            if (test_labels[j] == query_labels[i]
                    and test_cameras[j] == query_cameras[i]):
                continue
            vaild_num += 1
            if vaild_num > x:
                break
            if (test_labels[j] == query_labels[i]
                    and test_cameras[j] != query_cameras[i]):
                good = True
                break
        if good:
            rank_x += 1
        total += 1
    rank_x /= total
    return rank_x

def get_map(dist, query_labels, query_cameras, test_labels, test_cameras):
    indices = np.argsort(dist, axis=1)
    matches = (test_labels[indices] == query_labels[:, np.newaxis])
    m, _ = dist.shape
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        valid = ((test_labels[indices[i]] != query_labels[i]) |
                 (test_cameras[indices[i]] != query_cameras[i]))
        y_true = matches[i, valid]
        y_score = -dist[i][indices[i]][valid]
        if not np.any(y_true): continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    return float(np.sum(aps)) / np.sum(is_valid_query)

def visualize(dist, query_files, test_files):
    canvas = Image.new('RGB', (600, 1000), (255, 255, 255))
    idx = np.random.randint(0, len(dist), (10))
    rows = dist[idx]
    q_files = query_files[idx]
    for i, row in enumerate(rows):
        img = Image.open(q_files[i]).resize((50, 100))
        canvas.paste(img, (0, i*100))
        candidates = test_files[np.argsort(row)[:10]]
        for j, candidate in enumerate(candidates):
            img = Image.open(candidate).resize((50, 100))
            canvas.paste(img, (100+j*50, i*100))
    canvas.save('visualize.png')
    try:
        import matplotlib.pyplot as plt
        plt.imshow(np.asarray(canvas))
    except:
        log('[NOTE] Failed to show image by matplotlib.')

def test(args):

    feat_extractor = FeatureExtractor(state_path=args.model_file,
                                      last_conv=args.last_conv)
    if args.use_gpu:
        feat_extractor = DataParallel(feat_extractor)
        feat_extractor.cuda()

    feat_dim = 256 if args.last_conv else 2048

    log('[START] Loading Data')
    queryset = Market1501(args.dataset, data_type='query',
                          transform=transform, once=args.load_once)
    testset = Market1501(args.dataset, data_type='test',
                         transform=transform, once=args.load_once)
    queryloader = DataLoader(queryset, batch_size=args.batch_size,
                             num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    log('[ END ] Loading Query Data')

    log('[START] Extracting Query Features')
    query_feat, query_labels, query_cameras, query_files = extract_feat(
        args, feat_extractor, queryloader, feat_dim)
    log('[ END ] Extracting Query Features')

    log('[START] Extracting Test Features')
    test_feat, test_labels, test_cameras, test_files = extract_feat(
        args, feat_extractor, testloader, feat_dim)
    log('[ END ] Extracting Test Features')

    log('[START] Calculating Distances')
    dist = None
    if args.use_gpu:
        dist = calc_dist(query_feat, test_feat)
    else:
        dist = get_dist(query_feat, test_feat)
    log('[ END ] Calculating Distances')

    log('[START] Evaluating mAP, Rank-x')
    mAP = get_map(dist, query_labels, query_cameras,
                  test_labels, test_cameras)
    rank1 = get_rank_x(1, dist, query_labels, query_cameras,
                       test_labels, test_cameras)
    rank10 = get_rank_x(10, dist, query_labels, query_cameras,
                        test_labels, test_cameras)
    log('[ END ] Evaluating mAP, Rank-x')

    log('mAP: %f\trank-1: %f\trank-10: %f' % (mAP, rank1, rank10))
    visualize(dist, query_files, test_files)

    return mAP, rank1, rank10