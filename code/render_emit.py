#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append('../code')
from dataset.dataset import NeILFDataset
from dataset.dataset_fipt import FIPTDataset
from model.neilf_brdf import NeILFModel
from utils import general, io

def save_image(image, path, colormap=False):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image*255).astype(np.uint8)
    if colormap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(path)

def evaluate(input_data_folder,
             output_model_folder,
             config_path,
             timestamp,
             checkpoint,
             eval_nvs,
             eval_brdf,
             eval_lighting,
             export_nvs,
             export_brdf,
             export_lighting):

    assert os.path.exists(input_data_folder), "Data directory is empty"
    assert os.path.exists(output_model_folder), "Model directorty is empty"
    torch.set_default_dtype(torch.float32)

    # load config file
    config = io.load_config(config_path)

    # load input data and create evaluation dataset
    validation_indexes = config['eval']['validation_indexes']
    num_pixel_samples = config['train']['num_pixel_samples']
    dataset = config['dataset']

    eval_dataset = FIPTDataset(
            input_data_folder, validation_indexes, num_pixel_samples, dataset=dataset, mode='train')
    # image size
    H = eval_dataset.image_resolution[0]
    W = eval_dataset.image_resolution[1]
    num_val_images = len(eval_dataset.validation_indexes)

    # create model
    model = NeILFModel(config['model'])
    if torch.cuda.is_available():
        model.cuda()

    # load model
    if timestamp == 'latest':
        timestamps = os.listdir(output_model_folder)
        if len(timestamps) == 0:
            print('WRONG MODEL FOLDER')
            exit(-1)
        else:
            timestamp = sorted(timestamps)[-1]
    checkout_folder = os.path.join(output_model_folder, timestamp, 'checkpoints')
    prefix = str(checkpoint) + '.pth'
    model_path = os.path.join(checkout_folder, 'ModelParameters', prefix)
    model_params = torch.load(model_path)
    model.load_state_dict(model_params['model_state_dict'])
    neilf_pbr = model.neilf_pbr

    # create evaluation folder
    eval_folder = os.path.join(output_model_folder, timestamp, 'evaluation')
    general.mkdir_ifnotexists(eval_folder)

    # evaluate BRDFs and novel view renderings
    # results = dict()
    results = defaultdict(lambda: defaultdict(dict))

    # get validation data in the dataset
    model_input, ground_truth = eval_dataset.validation_data
    for attr in ['intrinsics', 'pose', 'uv', 'positions', 'normals']:
        model_input[attr] = model_input[attr].cuda()

    print('len of val data:', len(model_input['pose']))
    poses = model_input['pose']          # (n, 1, 4, 4)
    positions = model_input['positions'] # (n, H*W, 3)
    
    dir_out = {}
    split = 'train'
    for name in ['rgb', 'diffuse', 'a_prime', 'roughness', 'metallic', 'emission', 'merge']:
        d = Path(eval_folder) / split / name
        d.mkdir(exist_ok=True, parents=True)
        dir_out[name] = d

    normalization_factor = 0.0
    if 'bathroom' in input_data_folder:
        normalization_factor = 16
    elif 'bedroom' in input_data_folder:
        normalization_factor = 12
    elif 'kitchen' in input_data_folder:
        normalization_factor = 16
    elif 'livingroom' in input_data_folder: 
        normalization_factor = 18
    else:
        normalization_factor = 6
    
    # print('normalization_factor:', normalization_factor)
    for i in tqdm(range(num_val_images)):
        pose = poses[i][0]
        t = pose[:3, -1]
        t_map = t.view(1, 1, -1).repeat(H, W, 1)
        position = positions[i].view(H, W, -1)
        incident_dirs = F.normalize(t_map - position, dim=-1)
        emission = neilf_pbr.sample_incident_lights(t_map[:, 0, :], incident_dirs)
        emission = emission.detach().cpu().numpy() / normalization_factor


        path = dir_out['emission'] / '{:0>5d}_emission.png'.format(i)
        save_image(emission, path)
        


    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_data_folder', type=str, 
                        help='input folder of images, cameras and geometry files.')
    parser.add_argument('output_model_folder', type=str, 
                        help='folder containing trained models, and for saving results')
    parser.add_argument('--config_path', type=str, 
                        default='./confs/synthetic_neilf_brdf.json')    

    # checkpoint
    parser.add_argument('--timestamp', default='latest',
                        type=str, help='the timestamp of the model to be evaluated.')
    parser.add_argument('--checkpoint', default='latest',
                        type=str, help='the checkpoint of the model to be evaluated')
    
    # items to evaluate
    parser.add_argument('--eval_nvs', action='store_true', default=False, 
                        help="evaluate novel view renderings")
    parser.add_argument('--eval_brdf', action='store_true', default=False, 
                        help="evaluate BRDF maps at novel views")
    parser.add_argument('--eval_lighting', action='store_true', default=False, 
                        help="work in progress, not ready yet")
    parser.add_argument('--export_nvs', action='store_true', default=False, 
                        help="export novel view renderings")
    parser.add_argument('--export_brdf', action='store_true', default=False, 
                        help="export BRDF as texture maps")
    parser.add_argument('--export_lighting', action='store_true', default=False, 
                        help="export incident lights at certain positions")

    args = parser.parse_args()
        
    evaluate(args.input_data_folder,
             args.output_model_folder,
             args.config_path,
             args.timestamp,
             args.checkpoint,
             args.eval_nvs,
             args.eval_brdf,
             args.eval_lighting,
             args.export_nvs,
             args.export_brdf,
             args.export_lighting)