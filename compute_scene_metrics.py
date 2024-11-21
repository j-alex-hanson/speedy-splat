#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from scene import Scene
import csv
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def get_kernel_times(iteration, name, cameras, scene, renderFunc, renderArgs):

    repeat = 20
    print('dataset repeat: {repeat}')

    kernel_times = []
    pbar = tqdm(
        total=len(cameras) * repeat,
        desc=f"{name.capitalize()} Kernel Time Progress"
    )
    # Warm-up
    for viewpoint in cameras[:5]:
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

    for _ in range(repeat):
        for viewpoint in cameras:
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            kernel_times.append(render_pkg["kernel_times"])
            pbar.update(1)
    pbar.close()

    kernel_times = torch.stack(kernel_times)

    return kernel_times

def process_kernel_times(model_path, name, iteration, views, scene, renderFunc, renderArgs):

    output_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(output_path, exist_ok=True)

    kernel_times = get_kernel_times(iteration, name, views, scene,
                                renderFunc, renderArgs)

    kernel_times = kernel_times.detach().cpu().numpy()
    csv_path = os.path.join(output_path, "kernel_times.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Overall"])
        for k_t in kernel_times:
            writer.writerow(k_t)

def scene_metrics(iteration, name, cameras, scene, renderFunc, renderArgs):
    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    render_time_test = 0.0

    pbar = tqdm(
        total=len(cameras), desc=f"{name.capitalize()} Image Metric Progress")
    # Warm-up
    for viewpoint in cameras[:5]:
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

    for viewpoint in cameras:
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        kernel_times = render_pkg["kernel_times"]

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim(image, gt_image).mean().double()
        lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
        render_time_test += kernel_times[-1].item()

        pbar.update(1)

    pbar.close()
    l1_test /= len(cameras)
    psnr_test /= len(cameras)
    ssim_test /= len(cameras)
    lpips_test /= len(cameras)
    render_time_test /= len(cameras)

    fps_test = 1000 / render_time_test

    print("\n[ITER {}] Evaluation {}: \n\t L1 {} \n\t PSNR {} \n\t SSIM {} \n\t LPIPS {} \n\t FPS {}".format(
        iteration, name.capitalize(),
        l1_test, psnr_test, ssim_test, lpips_test, fps_test))

    return l1_test, psnr_test, ssim_test, lpips_test, fps_test


def compute_scene_metrics(model_path, name, iteration, views, scene, renderFunc, renderArgs):

    output_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(output_path, exist_ok=True)

    metrics_pkg = scene_metrics(iteration, name, views, scene,
                                renderFunc, renderArgs)
    l1, psnr, ssim, lpips, fps = metrics_pkg
    l1 = l1.item() if isinstance(l1, torch.Tensor) else l1
    psnr = psnr.item() if isinstance(psnr, torch.Tensor) else psnr
    ssim = ssim.item() if isinstance(ssim, torch.Tensor) else ssim
    lpips = lpips.item() if isinstance(lpips, torch.Tensor) else lpips

    csv_path = os.path.join(output_path, "metrics.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["l1", "psnr", "ssim", "lpips", "fps"])
        writer.writerow([l1, psnr, ssim, lpips, fps])


def run(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, suffix : str, kernel_times : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        metrics_func = process_kernel_times if kernel_times else compute_scene_metrics

        if not skip_train:
             metrics_func(dataset.model_path, f"train{suffix}", scene.loaded_iter, scene.getTrainCameras(), scene, render, (pipeline, background))

        if not skip_test:
             metrics_func(dataset.model_path, f"test{suffix}", scene.loaded_iter, scene.getTestCameras(), scene, render, (pipeline, background))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--kernel_times", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.suffix:
        args.suffix = f"_{args.suffix}"

    run(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.suffix, args.kernel_times)
