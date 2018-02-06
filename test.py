import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from net import FeatureExtractor
from data import Market1501
from config import transform
from config import conf
from utils import get_time

def extract_feat(extractor, dataloader, feat_dim):
    feat = []
    labels = []
    cameras = []
    for i, data in enumerate(dataloader):
        inputs, l, c = data
        inputs = Variable(inputs)
        if conf['use_gpu']:
            inputs = inputs.cuda()
        outputs = extractor.forward(inputs)
        feat.append(outputs)
        labels.append(l)
        cameras.append(c)
    feat = torch.cat(feat)
    feat.view(-1, feat_dim)
    labels = torch.cat(labels)
    cameras = torch.cat(cameras)
    return feat, labels, cameras

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

def test(last_conv=True):
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    dataset_path = os.path.join(torch_home, 'datasets')
    state_path = os.path.join(torch_home, 'models', conf['model_name'])

    print('%s [START] Loading Test Data' % get_time())
    queryset = Market1501(dataset_path, data_type='query', transform=transform)
    queryloader = DataLoader(queryset, batch_size=64)
    print('%s [ END ] Loading Test Data' % get_time())

    print('%s [START] Loading Query Data' % get_time())
    testset = Market1501(dataset_path, data_type='test', transform=transform)
    testloader = DataLoader(testset, batch_size=64)
    print('%s [ END ] Loading Query Data' % get_time())    

    feat_extractor = FeatureExtractor(state_path=state_path, last_conv=last_conv)

    feat_dim = 2048
    if last_conv: feat_dim = 256
    
    print('%s [START] Extracting Query Features' % get_time())
    query_feat, query_labels, query_cameras = extract_feat(feat_extractor, queryloader, feat_dim)
    print('%s [ END ] Extracting Query Features' % get_time())
    print('%s [START] Extracting Test Features' % get_time())
    test_feat, test_labels, test_cameras = extract_feat(feat_extractor, testloader, feat_dim)
    print('%s [ END ] Extracting Test Features' % get_time())

    print('%s [START] Extracting Test Features' % get_time())
    dist = get_dist(query_feat, test_feat)
    print('%s [ END ] Extracting Test Features' % get_time())

    print('%s [START] Evaluating mAP, Rank-x' % get_time())
    mAP = get_map(dist, query_labels, query_cameras, test_labels, test_cameras)
    rank1 = get_rank_x(1, dist, query_labels, query_cameras, test_labels, test_cameras)
    rank10 = get_rank_x(10, dist, query_labels, query_cameras, test_labels, test_cameras)
    print('%s [ END ] Evaluating mAP, Rank-x' % get_time())

    return mAP, rank1, rank10