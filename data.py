import os
import re
import torch
from PIL import Image
import torch.utils.data as data

from utils import log

class Market1501(data.Dataset):

    base_folder = 'Market-1501-v15.09.15'
    train_folder = 'test'
    train_folder = 'bounding_box_train'
    test_folder = 'bounding_box_test'
    query_folder = 'query'

    def __init__(self, root, data_type='train',
                transform=None, target_transform=None,
                download=False, once=False):
        self.root = root
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.once = once

        if download:
            self.download()

        if self.data_type == 'train':
            self.folder = os.path.join(self.root, self.base_folder, self.train_folder)
        elif self.data_type == 'test':
            self.folder = os.path.join(self.root, self.base_folder, self.test_folder)
        else:
            self.folder = os.path.join(self.root, self.base_folder, self.query_folder)

        self.pattern = re.compile(r'^(\-1|\d{4})_c(\d)s\d_\d{6}_\d{2}.*\.jpg$')
        self.file_list = list(filter(self.pattern.search, os.listdir(self.folder)))

        if self.once:
            self.load_data_at_once()

    def load_data_at_once(self):
        self.data, self.labels, self.cameras = [], [], []
        k, total = 0, len(self.file_list)
        for file in self.file_list:
            img, label, camera = self.load_image(file)
            self.data.append(img)
            self.labels.append(label)
            self.cameras.append(camera)

            if k % 500 == 499:
                log('[%s_data_loading] %5d/%5d' % (self.data_type, k, total))
            k += 1

        self.data = torch.cat(self.data, 0)
        self.data = self.data.view(-1, 3, 768, 256)

    def __getitem__(self, index):
        if self.once:
            return self.data[index], self.labels[index], self.cameras[index]
        return self.load_image(self.file_list[index])

    def load_image(self, filename):
        label, camera = re.findall(self.pattern, filename)[0]
        label, camera = int(label), int(camera)
        img = Image.open(os.path.join(self.folder, filename))
        img.load()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img = self.target_transform(img)

        return img, label, camera, filename

    def __len__(self):
        return len(self.file_list)

    def download(self):
        pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str
