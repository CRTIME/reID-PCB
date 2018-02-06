import os
import re
import torch
from utils import get_time
from PIL import Image
import torch.utils.data as data

class Market1501(data.Dataset):
    
    base_folder = 'Market-1501-v15.09.15'
    train_folder = 'test'
    train_folder = 'bounding_box_train'
    test_folder = 'bounding_box_test'
    query_folder = 'query'
    
    def __init__(self, root, data_type='train',
                transform=None, target_transform=None,
                download=False):
        self.root = root
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
        
        if self.data_type == 'train':
            folder = os.path.join(self.root, self.base_folder, self.train_folder)
            self.train_data, self.train_label, self.train_camera, self.train_size = self.load_data(folder)
        elif self.data_type == 'test':
            folder = os.path.join(self.root, self.base_folder, self.test_folder)
            self.test_data, self.test_label, self.test_camera, self.test_size = self.load_data(folder)
        else:
            folder = os.path.join(self.root, self.base_folder, self.query_folder)
            self.query_data, self.query_label, self.query_camera, self.query_size = self.load_data(folder)

    def load_data(self, folder):

        file_list = os.listdir(folder)
        pattern = re.compile(r'^(\-1|\d{4})_c(\d)s\d_\d{6}_\d{2}\..+$')

        data = []
        labels = []
        cameras = []

        k = 0
        total = len(file_list)
        data_size = 0
        label_dict = {}
        
        for file in file_list:
            re_result = re.findall(pattern, file)
            if not len(re_result) == 1:
                print('%s [WARRNING]: inopportune file %s in %s' % (get_time(), file, folder))
                continue
            label, camera = re_result[0]
            label, camera = int(label), int(camera)
            img = self.load_image(os.path.join(folder, file))

            if label not in label_dict:
                label_dict[label] = int(data_size)
                data_size += 1
            if self.data_type == 'train':
                label = label_dict[label]

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                img = self.target_transform(img)
            
            data.append(img)
            labels.append(label)
            cameras.append(camera)

            if k % 500 == 499:
                print('%s [%s-data-loading] %3.3f%% [data-size] %4d' % (get_time(), self.data_type, 100 * k / total, data_size))                
            k += 1

        data = torch.cat(data, 0)
        data = data.view(-1, 3, 768, 256)

        return data, labels, cameras, data_size
        
    def __getitem__(self, index):
        if self.data_type == 'train':
            img = self.train_data[index]
            label = self.train_label[index]
            camera = self.train_camera[index]
        elif self.data_type == 'test':
            img = self.test_data[index]
            label = self.test_label[index]
            camera = self.test_camera[index]
        else:
            img = self.query_data[index]
            label = self.query_label[index]
            camera = self.query_camera[index]
        return img, label, camera
        
    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_data)
        elif self.data_type == 'test':
            return len(self.test_data)
        else:
            return len(self.query_data)
        
    def download(self):
        pass
    
    def load_image(self, filename) :
        img = Image.open(filename)
        img.load()
        return img
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str
