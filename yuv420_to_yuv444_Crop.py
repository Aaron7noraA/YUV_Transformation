import os
import numpy as np
import argparse
import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
import torch.nn.functional as F

from Dataset import DATASETS


def conversion(source_path, dst_path, Shape):
    a_channel_slice = size = np.prod(Shape)
    source = open(source_path, 'rb')
    with open(dst_path, 'wb') as dest:
        num_frames = os.path.getsize(source_path) // (a_channel_slice + a_channel_slice//2)
        for _ in range(num_frames):
            Y = source.read(a_channel_slice)
            Y = np.frombuffer(Y, dtype=np.uint8).copy().reshape(1, Shape[0], Shape[1])
            Y = torch.from_numpy(Y)
            Y = transform.CenterCrop((1024, 1920))(Y).numpy().tobytes()
            
            U = source.read(a_channel_slice//4)
            U = np.frombuffer(U, dtype=np.uint8).copy().reshape(1, Shape[0]//2, Shape[1]//2)
            U = torch.from_numpy(U)
            U = F.interpolate(U.unsqueeze(0), scale_factor=2, mode='nearest')[0]
            U = transform.CenterCrop((1024, 1920))(U).numpy().tobytes()
            
        

            V = source.read(a_channel_slice//4)
            V = np.frombuffer(V, dtype=np.uint8).copy().reshape(1, Shape[0]//2, Shape[1]//2)
            V = torch.from_numpy(V)
            V = F.interpolate(V.unsqueeze(0), scale_factor=2, mode='nearest')[0]
            V = transform.CenterCrop((1024, 1920))(V).numpy().tobytes()


            dest.write(Y)
            dest.write(U)
            dest.write(V)

if __name__ == '__main__':

    source_root = '/work/u6716795/video_dataset/TestVideo/raw_video_1080/'
    dst_path_root = '/work/u6716795/YUV_Testing_Dataset/YUV444_C'

    Shape = (1080, 1920)

    for dataset, seq_list in DATASETS.items():
        
        os.makedirs(os.path.join(dst_path_root, dataset), exist_ok=True)
        for seq in seq_list.keys():
            os.makedirs(os.path.join(dst_path_root, dataset, seq), exist_ok=True)
            source = os.path.join(source_root, dataset, DATASETS[dataset][seq]['vi_name'])
            dst_path = os.path.join(dst_path_root, dataset, seq, DATASETS[dataset][seq]['vi_name'])
            
            conversion(source, dst_path, Shape)