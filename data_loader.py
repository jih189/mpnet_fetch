import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import skimage.io

import pickle
import re
import numpy as np
import open3d as o3d

import os
from os import path as osp

import torch_geometric.data as tg_data

class MPNetDataLoader(Dataset):
    ''' Custom dataset object for training the MPNet model.
    '''

    def __init__(self, data_folder, env_list, max_point_cloud_size, q_min_, q_max_):
        '''
        :param data_folder: location of where file exists.
        '''
        self.data_folder = data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                           for filei in os.listdir(osp.join(data_folder, f'env_{envNum:06d}'))
                           if filei.endswith('.p')
                           ]

        # Keeping env information fixed.
        self.max_point_cloud_size = max_point_cloud_size

        # Set the joint limit
        self.q_min = q_min_
        self.q_max = q_max_ 

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)*2

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        env_num, path_num = self.index_dict[index//2]
        envFolder = osp.join(self.data_folder, f'env_{env_num:06d}')

        # Load the pcd data.
        env_data_folder = osp.join(self.data_folder, f'env_{env_num:06d}')
        map_file = osp.join(env_data_folder, f'map_{env_num}.ply')
        data_PC = o3d.io.read_point_cloud(map_file, format='ply')
        total_number_points = np.array(data_PC.points).shape[0]
        ratio = min((1, (self.max_point_cloud_size+1)/total_number_points))
        down_sample_PC = data_PC.random_down_sample(ratio)
        depth_points = np.array(down_sample_PC.points)[:self.max_point_cloud_size, :]

        #  Load the path
        with open(osp.join(envFolder, f'path_{path_num}.p'), 'rb') as f:
            data_path = pickle.load(f, encoding='latin1')
            joint_path = data_path['path']
        # Normalize the trajectory.
        q = 2*(joint_path-self.q_min)/(self.q_max-self.q_min) - 1

        # set goal position.
        input_pos = np.concatenate([q[:-1, :7], np.ones_like(q[:-1, :7])*q[-1, :7]], axis=1)
        return {
            'input_pos': torch.as_tensor(input_pos, dtype=torch.float),
            'target_pos': torch.as_tensor(q[1:, :7], dtype=torch.float),
            'env': torch.as_tensor(depth_points.reshape(-1), dtype=torch.float)
        }

def get_mpnet_padded_seq(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['input_pos'] = pad_sequence(
        [batch_i['input_pos'] for batch_i in batch], batch_first=True
    )
    data['target_pos'] = pad_sequence(
        [batch_i['target_pos'] for batch_i in batch], batch_first=True
    )
    data['mask'] = pad_sequence([torch.ones(batch_i['input_pos'].shape[0])
                                for batch_i in batch], batch_first=True)
    data['env'] = torch.cat([batch_i['env'][None, :] for batch_i in batch], dim=0)
    return data

def main():
    # the load dataset test
    q_min_fetch = np.array([[-1.6056, -1.221,-np.pi, -2.251, -np.pi, -2.16, -np.pi]])
    q_max_fetch = np.array([[1.6056, 1.518, np.pi, 2.251, np.pi, 2.16, np.pi]])
    data_loader = MPNetDataLoader('/root/trajectory_data_with_constraints', list(range(2)), 2000, q_min_fetch, q_max_fetch)
    data = data_loader[0]
    print(data['input_pos'].size())
    print(data['target_pos'].size())
    print(data['env'].size())
    print("load done")

if __name__ == "__main__":
    main()
