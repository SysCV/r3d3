import torch
import cv2
import lietorch
import r3d3_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import r3d3.geom.projective_ops as pops
import os

CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [1, -1, 1.5],
    [1, 1, 1.5],
    [-1, 1, 1.5],
    [-0.5, 1, 1.5],
    [0.5, 1, 1.5],
    [0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_traj_actor(traj, g):
    traj_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
    )
    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    traj_actor.paint_uniform_color(color)
    return traj_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.traj = None
    droid_visualization.traj_points = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 205  # 0.1 #0.005
    droid_visualization.mode = 'run'

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def animation_callback(vis):
        add_camera = False
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value
                dirty_index_flat, = torch.where(video.dirty_flat.clone())
                dirty_index, = torch.where(video.dirty.clone())

            if len(dirty_index_flat) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            disps = torch.index_select(video.disps_flat, 0, dirty_index_flat)

            main_Ps = SE3(torch.index_select(video.poses, 0, dirty_index)).inv().matrix().cpu().numpy()
            poses = torch.index_select((SE3(video.rel_poses) * SE3(video.poses.unsqueeze(1))).view((-1,)).data, 0,
                                       dirty_index_flat)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images_flat, 0, dirty_index_flat)
            images = images.cpu()[:, [0, 1, 2], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
            points = r3d3_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics_flat).cpu()  # ToDo intr.

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))  # ToDo intr.

            count = r3d3_backends.depth_filter(
                poses, video.disps_flat, video.intrinsics_flat, dirty_index_flat, thresh)

            count = count.cpu()
            disps = disps.cpu()
            occl_masks = video.masks[dirty_index_flat % video.n_cams].cpu()
            masked_disps_mean = (disps * occl_masks).sum([1, 2], keepdims=True) / occl_masks.sum([1, 2], keepdims=True)
            masks = ((count >= 2) & (disps > .5 * masked_disps_mean) & occl_masks)

            for i in range(len(dirty_index)):
                ix = dirty_index[i].item()
                if droid_visualization.traj is not None:
                    # with droid_visualization.video.get_lock():
                    #    print("Test")
                    # vis.remove_geometry(droid_visualization.traj)
                    del droid_visualization.traj

                if ix in droid_visualization.traj_points:
                    del droid_visualization.traj_points[ix]

                ### add trajectory actor ###
                droid_visualization.traj_points[ix] = main_Ps[i, 0:3, 3]
                idcs = list(droid_visualization.traj_points.keys())
                if len(idcs) > 1:
                    idcs.sort()
                    traj_actor = create_traj_actor(np.stack([droid_visualization.traj_points[i] for i in idcs]), True)
                    vis.add_geometry(traj_actor)
                    droid_visualization.traj = traj_actor

            for i in range(len(dirty_index_flat)):
                pose = Ps[i]

                ix = dirty_index_flat[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True, scale=0.01)
                cam_actor.transform(pose)
                if add_camera:
                    vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("configs/misc/renderoption.json")

    vis.run()
    vis.destroy_window()
