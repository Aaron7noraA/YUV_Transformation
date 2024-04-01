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

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

def rgb_to_ycbcr420(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)
    

    return np.concatenate((y, uv), axis=0)

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
            img = rgb_to_ycbcr420(img)
            img = img * 255
            
            img = torch.from_numpy(img)
            img = img.numpy().astype('uint8').tobytes()
            
            dest.write(img)

def worker(args):
    return conversion(args)

if __name__ == '__main__':

    source_root = '/work/u6716795/BT709/'
    dst_path_root = '/work/u6716795/YUV_Testing_Dataset/BT709_YUV444_Quantize'


    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=30)
    # print(os.path.getsize('/work/u6716795/YUV_Testing_Dataset/BT709_YUV444_C/HEVC-B/Cactus/Cactus_1920x1080_50.yuv'))
    # exit()
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
            # conversion(source, dst_path, DATASETS[dataset][seq]['frameNum'])