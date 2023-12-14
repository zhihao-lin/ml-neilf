#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
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
    # split inputs
    total_pixels = eval_dataset.total_pixels
    split_inputs = general.split_neilf_input(model_input, total_pixels)

    # generate outputs
    split_outputs = []
    for i in tqdm(range(len(split_inputs))):
        split_input = split_inputs[i]
        with torch.no_grad():
            split_output = model(split_input, is_training=False)
        split_outputs.append(
            {k:split_output[k].detach().cpu() for k in 
            ['rgb_values', 'points', 'normals', 'base_color',
            'roughness', 'metallic', 'render_masks']})

    # merge output
    num_val_images = len(eval_dataset.validation_indexes)
    model_outputs = general.merge_neilf_output(
        split_outputs, total_pixels, num_val_images)

    # image size
    H = eval_dataset.image_resolution[0]
    W = eval_dataset.image_resolution[1]

    # rendered mask
    mask = model_outputs['render_masks']
    mask = mask.reshape([num_val_images, H, W, 1]).float()
    mask_np = mask.numpy()

    # estimated image
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape([num_val_images, H, W, 3])
    if not config['model']['use_ldr_image']: 
        rgb_eval = general.hdr2ldr(rgb_eval)
    rgb_eval = rgb_eval * mask + (1 - mask)
    rgb_eval_np = rgb_eval.numpy()


    # estimated BRDF
    base_eval = model_outputs['base_color'].reshape([num_val_images, H, W, 3])    
    base_eval = base_eval * mask + (1 - mask)
    roug_eval = model_outputs['roughness'].reshape([num_val_images, H, W, 1])    
    meta_eval = model_outputs['metallic'].reshape([num_val_images, H, W, 1])
    base_eval_np = base_eval.numpy()
    roug_eval_np = roug_eval.numpy()
    meta_eval_np = meta_eval.numpy()

    dir_out = {}
    split = 'train'
    for name in ['rgb', 'diffuse', 'a_prime', 'roughness', 'metallic', 'emission', 'merge']:
        d = Path(eval_folder) / split / name
        d.mkdir(exist_ok=True, parents=True)
        dir_out[name] = d
    
    for i in tqdm(range(num_val_images)):
        rgb = rgb_eval_np[i]
        path = dir_out['rgb'] / '{:0>5d}_rgb.png'.format(i)
        save_image(rgb, path)
        
        albedo = base_eval_np[i]
        roughness = roug_eval_np[i]
        metallic = meta_eval_np[i]

        kd = albedo * (1 - metallic)
        path = dir_out['diffuse'] / '{:0>5d}_kd.png'.format(i)
        save_image(kd, path)

        ks = 0.04*(1-metallic) + albedo*metallic
        a_prime = ks + kd 
        path = dir_out['a_prime'] / '{:0>5d}_a_prime.png'.format(i)
        save_image(a_prime, path)
        
        path = dir_out['roughness'] / '{:0>5d}_roughness_color.png'.format(i)
        save_image(roughness, path, colormap=True)
        roughness = roughness.repeat(3, -1)
        path = dir_out['roughness'] / '{:0>5d}_roughness.png'.format(i)
        save_image(roughness, path)

        path = dir_out['metallic'] / '{:0>5d}_metallic_color.png'.format(i)
        save_image(metallic, path, colormap=True)
        metallic = metallic.repeat(3, -1)
        path = dir_out['metallic'] / '{:0>5d}_metallic.png'.format(i)
        save_image(metallic, path)
        


    
    

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