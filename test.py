import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from utils import log
from net import FeatureExtractor
from data import Market1501
from config import transform
from utils import get_time

def extract_feat(args, extractor, dataloader, feat_dim):
    feat = []
    labels = []
    cameras = []
    for i, data in enumerate(dataloader):
        inputs, l, c = data
        inputs = Variable(inputs, volatile=True)
        if args.use_gpu:
            inputs = inputs.cuda()
        outputs = extractor.forward(inputs)
        feat.append(outputs)
        labels.append(l)
        cameras.append(c)
    feat = torch.cat(feat)
    feat.view(-1, feat_dim)
    labels = torch.cat(labels)
    cameras = torch.cat(cameras)
    return feat.cpu().data.numpy(), labels.numpy(), cameras.numpy()

def get_dist(query, test):
    return cdist(query, test)

def get_rank_x(x, dist, query_labels, query_cameras, test_labels, test_cameras):
    rank_x = 0
    for i, row in enumerate(dist):
        index = np.argsort(row)[:-x-1:-1]
        good = 0
        for j in index:
            if test_labels[j] == query_labels[i] and test_cameras[j] != query_cameras[i]:
                good += 1
        rank_x += good / len(index)
    rank_x /= np.shape(dist)[0]
    return rank_x

def get_map(dist, query_labels, query_cameras, test_labels, test_cameras):
    mAP = 0
    for i, row in enumerate(dist):
        index = np.argsort(row)[::-1]
        ap, good, total = 0, 0, 0
        for j in index:
            total += 1
            if test_labels[j] == query_labels[i] and test_cameras[j] != query_cameras[i]:
                good += 1
                ap += good / total
        ap /= good
        mAP += ap
    mAP /= np.shape(dist)[0]
    return mAP

def print_result(mAP, rank1, rank10, dist, query_labels, query_cameras, test_labels, test_cameras):
    print('mAP: %f\trank-1: %f\trank-10: %f' % (mAP, rank1, rank10))
    f = open('test_result.txt', 'a+')
    for i, row in enumerate(dist):
        ql, qc = query_labels[i], query_cameras[i]
        s = '[%d,%d]' % (ql, qc)
        index = np.argsort(row)[::-1]
        for j in index:
            tl, tc = test_labels[j], test_cameras[j]
            s += '\t[%d,%d]' % (tl, tc)
        s += '\n'
        f.write(s)
    f.close()

def test(args):

    feat_extractor = FeatureExtractor(state_path=args.model_file, last_conv=args.last_conv)
    if args.use_gpu:
        feat_extractor.cuda()

    feat_dim = 2048
    if args.last_conv: feat_dim = 256

    log('[START] Loading Query Data')
    queryset = Market1501(args.dataset, data_type='query', transform=transform, once=args.load_once)
    queryloader = DataLoader(queryset, batch_size=args.batch_size, num_workers=args.num_workers)
    log('[ END ] Loading Query Data')

    log('[START] Extracting Query Features')
    query_feat, query_labels, query_cameras = extract_feat(args, feat_extractor, queryloader, feat_dim)
    log('[ END ] Extracting Query Features')

    log('[START] Loading Test Data')
    testset = Market1501(args.dataset, data_type='test', transform=transform, once=args.load_once)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    log('[ END ] Loading Test Data')

    log('[START] Extracting Test Features')
    test_feat, test_labels, test_cameras = extract_feat(args, feat_extractor, testloader, feat_dim)
    log('[ END ] Extracting Test Features')

    log('[START] Extracting Test Features')
    dist = get_dist(query_feat, test_feat)
    log('[ END ] Extracting Test Features')

    log('[START] Evaluating mAP, Rank-x')
    mAP = get_map(dist, query_labels, query_cameras, test_labels, test_cameras)
    rank1 = get_rank_x(1, dist, query_labels, query_cameras, test_labels, test_cameras)
    rank10 = get_rank_x(10, dist, query_labels, query_cameras, test_labels, test_cameras)
    log('[ END ] Evaluating mAP, Rank-x')

    return mAP, rank1, rank10