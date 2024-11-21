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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from compute_scene_metrics import scene_metrics

def score_func(view, gaussians, pipeline, background, scores):

    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render(view, gaussians, pipeline, background,
                   scores=img_scores)['render']

    # Backward computes and stores grad squared values
    # in img_scores's grad
    image.sum().backward()

    scores += img_scores.grad


def prune(scene, gaussians, pipe, background, prune_ratio):

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    torch.cuda.reset_peak_memory_stats()

    iter_start.record()

    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity)
        for view in scene.getTrainCameras():
            score_func(view, gaussians, pipe, background,
                scores)
            pbar.update(1)
        pbar.close()

    gaussians.prune_gaussians(prune_ratio, scores)

    iter_end.record()

    # Track peak memory usage (in bytes) and convert to MB
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    time_ms = iter_start.elapsed_time(iter_end)
    time_min = time_ms / 60_000

    return {
        "peak_memory_allocated" : peak_memory_allocated,
        "peak_memory_reserved" : peak_memory_reserved,
        "time_min" : time_min
    }

def training(dataset, opt, pipe, testing_iterations, visualize_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_time_ms = 0
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    prune_time_min = 0
    prune_peak_memory_allocated = 0
    prune_peak_memory_reserved = 0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        torch.cuda.reset_peak_memory_stats()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()
        # Track peak memory usage (in bytes) and convert to MB
        peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)


        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # --- Soft Pruning ---
                if (iteration >= opt.prune_from_iter) and \
                    (iteration < opt.prune_until_iter) and \
                    (iteration % opt.prune_interval == 0):

                    prune_pkg = prune(
                        scene, gaussians, pipe, background,
                        opt.densify_prune_ratio)
                    prune_time_min += prune_pkg['time_min']
                    prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                    prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # --- Hard Pruning ---
            if (iteration >= opt.densify_until_iter) and \
                (iteration >= opt.prune_from_iter) and \
                (iteration < opt.prune_until_iter) and \
                (iteration % opt.prune_interval == 0):

                prune_pkg = prune(
                    scene, gaussians, pipe, background,
                    opt.after_densify_prune_ratio)
                prune_time_min += prune_pkg['time_min']
                prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']


            # Log and save
            iter_time = iter_start.elapsed_time(iter_end)
            train_time_ms += iter_time
            train_time_min = train_time_ms / 60_000

            training_report(
                tb_writer, iteration, Ll1, loss,
                iter_time, train_time_min, prune_time_min,
                testing_iterations, visualize_iterations,
                peak_memory_allocated, peak_memory_reserved,
                prune_peak_memory_allocated, prune_peak_memory_reserved,
                scene, render,
                (pipe, background))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, iteration, Ll1, loss,
        iter_time, train_time, prune_time,
        testing_iterations, visualize_iterations,
        peak_memory_allocated, peak_memory_reserved,
        prune_peak_memory_allocated, prune_peak_memory_reserved,
        scene : Scene, renderFunc, renderArgs):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('time/iter_time', iter_time, iteration)
        tb_writer.add_scalar('time/train_time_minutes', train_time, iteration)
        tb_writer.add_scalar('time/prune_time_minutes', prune_time, iteration)
        tb_writer.add_scalar('counts/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('memory/peak_allocated_MB', peak_memory_allocated, iteration)
        tb_writer.add_scalar('memory/peak_reserved_MB', peak_memory_reserved, iteration)
        if prune_peak_memory_allocated > 0:
            tb_writer.add_scalar('memory/prune_peak_allocated_MB', prune_peak_memory_allocated, iteration)
        if prune_peak_memory_reserved > 0:
            tb_writer.add_scalar('memory/prune_peak_reserved_MB', prune_peak_memory_reserved, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        print("\n[ITER {}] Training Time: {} minutes".format(
            iteration, train_time))

        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)] }
        )

        for config in validation_configs:
            name, cameras = config['name'], config['cameras']
            if cameras and len(cameras) > 0:
                metrics = scene_metrics(iteration, name, cameras,
                    scene, renderFunc, renderArgs)
                if tb_writer:
                    tb_writer.add_scalar(
                        f'metrics_{name}/L1 Loss', metrics[0], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/PSNR', metrics[1], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/SSIM', metrics[2], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/LPIPS', metrics[3], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/FPS', metrics[4], iteration)

    if (iteration in visualize_iterations) and tb_writer:
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)] }
        )
        for config in validation_configs:
            for viewpoint in config['cameras'][:5]:
                image = torch.clamp(
                    renderFunc(
                        viewpoint, scene.gaussians, *renderArgs
                    )["render"],
                    0.0, 1.0)
                gt_image = torch.clamp(
                    viewpoint.original_image.to("cuda"),
                    0.0, 1.0)

                tb_writer.add_images(
                    config['name'] + "_view_{}/render".format(
                        viewpoint.image_name),
                    image[None], global_step=iteration)
                tb_writer.add_images(
                    config['name'] + "_view_{}/ground_truth".format(
                        viewpoint.image_name),
                    gt_image[None], global_step=iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--visualize_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.visualize_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
