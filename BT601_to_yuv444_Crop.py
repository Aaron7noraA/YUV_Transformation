import os
import numpy as np
import argparse
import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
import concurrent.futures

from Dataset import DATASETS
from torchvision.datasets.folder import default_loader as imgloader


def rgb_to_ycbcr420(RGB):
    T = torch.FloatTensor([[ 0.257,  0.504,   0.098],
                        [-0.148, -0.291,   0.439],
                        [ 0.439, -0.368,  -0.071]]).to(RGB.device)
    # T = torch.FloatTensor([[ 0.299,  0.587,   0.114],
    #                     [-0.168736, -0.331264,   0.5],
    #                     [ 0.5, -0.418668,  -0.081312]]).to(RGB.device)
    YUV = T.expand(RGB.size(0), -1, -1).bmm(RGB.flatten(2)).view_as(RGB)
    YUV[:, 0:1] = YUV[:, 0:1] + (16/256)
    YUV[:, 1:] = YUV[:, 1:] + (128/256)

    return YUV.clamp(min=0, max=1)

def cbcr420_to_rgb(y, uv):
    y = y - (16/256)
    uv = uv - (128/256)
    yuv = torch.cat([y, uv], dim=1)
    
    T = torch.FloatTensor([[ 0.257,  0.504,   0.098],
                        [-0.148, -0.291,   0.439],
                        [ 0.439, -0.368,  -0.071]]).to(y.device)
    T = torch.linalg.inv(T)
    rgb = T.expand(yuv.size(0), -1, -1).bmm(yuv.flatten(2)).view_as(yuv)

    return rgb.clamp(min=0, max=1)

def conversion(args):
    source_path = args['source_path']
    dst_path = args['dst_path']
    frameNum = args['frameNum']
    
    with open(dst_path, 'wb') as dest:
        for i in range(1, frameNum+1):
            img_name = os.path.join(source_path, f'frame_{i}.png')
            
            rgb = Image.open(img_name).convert('RGB')
            img = np.asarray(rgb).transpose(2, 0, 1)
            img = img / 255.
            img = torch.from_numpy(img).to(torch.float).unsqueeze(0)
            img = rgb_to_ycbcr420(img)[0]
            img = img * 255
            
            
            img = transform.CenterCrop((1024, 1920))(img).numpy().astype('uint8').tobytes()
            
            dest.write(img)
            
def worker(args):
    return conversion(args)

if __name__ == '__main__':

    source_root = '/work/u6716795/video_dataset/TestVideo/raw_video_1080/'
    dst_path_root = '/work/u6716795/YUV_Testing_Dataset/BT601_YUV444_C'

    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=120)

    for dataset, seq_list in DATASETS.items():
        
        os.makedirs(os.path.join(dst_path_root, dataset), exist_ok=True)
        for seq in seq_list.keys():
            os.makedirs(os.path.join(dst_path_root, dataset, seq), exist_ok=True)
            source = os.path.join(source_root, dataset, seq)
            dst_path = os.path.join(dst_path_root, dataset, seq, DATASETS[dataset][seq]['vi_name'])
            
            
            parse_msg = {}
            parse_msg['source_path'] = source
            parse_msg['dst_path'] = dst_path
            parse_msg['frameNum'] = DATASETS[dataset][seq]['frameNum']
            
            obj = threadpool_executor.submit(worker, parse_msg)