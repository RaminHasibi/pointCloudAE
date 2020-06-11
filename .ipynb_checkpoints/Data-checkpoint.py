import os
import os.path as osp
import shutil
import json
import warnings
import torch

from multiprocessing import Pool
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from external import PlyData
import numpy as np
import re
class ShapeNet_2048(InMemoryDataset):

    url = ('https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    # seg_classes = {
    #     'Airplane': [0, 1, 2, 3],
    #     'Bag': [4, 5],
    #     'Cap': [6, 7],
    #     'Car': [8, 9, 10, 11],
    #     'Chair': [12, 13, 14, 15],
    #     'Earphone': [16, 17, 18],
    #     'Guitar': [19, 20, 21],
    #     'Knife': [22, 23],
    #     'Lamp': [24, 25, 26, 27],
    #     'Laptop': [28, 29],
    #     'Motorbike': [30, 31, 32, 33, 34, 35],
    #     'Mug': [36, 37],
    #     'Pistol': [38, 39, 40],
    #     'Rocket': [41, 42, 43],
    #     'Skateboard': [44, 45, 46],
    #     'Table': [47, 48, 49],
    # }

    def __init__(self, root, categories=None, include_normals=True, transform=None, pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super(ShapeNet_2048, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        path = self.processed_paths[0]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ['train_test_split']

    @property
    def processed_file_names(self):
        cats = '_'.join([cat.lower() for cat in self.categories])
        return [os.path.join('{}.pt'.format(cats))]

    def load_ply(self,file_name, with_faces=False, with_color=False):
        ply_data = PlyData.read(file_name)
        points = ply_data['vertex']
        points = np.vstack([points['x'], points['y'], points['z']]).T
        ret_val = [points]

        if with_faces:
            faces = np.vstack(ply_data['face']['vertex_indices'])
            ret_val.append(faces)

        if with_color:
            r = np.vstack(ply_data['vertex']['red'])
            g = np.vstack(ply_data['vertex']['green'])
            b = np.vstack(ply_data['vertex']['blue'])
            color = np.hstack((r, g, b))
            ret_val.append(color)

        if len(ret_val) == 1:  # Unwrap the list
            ret_val = ret_val[0]

        return ret_val

    def files_in_subdirs(self, top_dir, search_pattern):
        regex = re.compile(search_pattern)
        for path, _, files in os.walk(top_dir):
            for name in files:
                full_name = osp.join(path, name)
                if regex.search(full_name):
                    yield full_name

    # def download(self):
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.root)
    #     # os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     name = self.url.split('/')[-1].split('.')[0]
    #     os.rename(osp.join(self.root, name), self.raw_dir)

    def pc_loader(self,f_name):
        ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme:
        i.e. /syn_id/model_name.ply
        '''
        tokens = f_name.split('/')
        model_id = tokens[-1].split('.')[0]
        synet_id = tokens[-2]
        return torch.tensor(self.load_ply(f_name)), model_id, synet_id

    def process_filenames(self, file_names, verbose=True):
        data_list = []
        pc = self.load_ply(file_names[0])
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}
        pclouds = torch.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=torch.float32)
        model_names = np.empty([len(file_names)], dtype=object)
        class_ids = np.empty([len(file_names)], dtype=object)
        pool = Pool(10)

        for i, data in enumerate(pool.imap(self.pc_loader, file_names)):
            x, model_names[i], class_ids[i] = data
            data = Data(x=x, category=cat_idx[class_ids[i]])
            data_list.append(data)

        pool.close()
        pool.join()

        if len(np.unique(model_names)) != len(pclouds):
            warnings.warn('Point clouds with the same model name were loaded.')

        if verbose:
            print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds),
                                                                                      len(np.unique(class_ids))))
        return data_list



    def process(self):
        # for i, split in enumerate(['train', 'val', 'test']):
        #     path = osp.join(self.raw_dir, 'train_test_split',
        #                     f'shuffled_{split}_file_list.json')
        data_list = []
        for cat in self.categories:
            path = osp.join(self.raw_dir, self.category_ids[cat])
            file_names = [f for f in self.files_in_subdirs(path, '.ply')]
            data_list+=self.process_filenames(file_names)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)

if __name__ == '__main__':
    train_dataset = ShapeNet_2048('../data/shapenet_2048' ,categories='Chair')
    print(train_dataset)