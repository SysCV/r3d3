from typing import Optional, List, Tuple
import numpy.typing as npt

import torch
import numpy as np

from shapely.geometry import Polygon


def graph_to_edge_list(graph, device='cpu'):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii, device=device)
    jj = torch.as_tensor(jj, device=device)
    kk = torch.as_tensor(kk, device=device)
    return ii, jj, kk


def multi_cam_graph_to_edge_list(graph, n_cams, device='cpu'):
    ii, jj, cii, cjj, kk = [], [], [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u[0])
            jj.append(v[0])
            cii.append(n_cams * u[0] + u[1])
            cjj.append(n_cams * v[0] + v[1])
            kk.append(s)

    cii = torch.as_tensor(cii, device=device)
    cjj = torch.as_tensor(cjj, device=device)
    ii = torch.as_tensor(ii, device=device)
    jj = torch.as_tensor(jj, device=device)
    kk = torch.as_tensor(kk, device=device)
    return ii, jj, cii, cjj, kk


def keyframe_indicies(graph):
    return torch.as_tensor([u for u in graph])


def meshgrid(m, n, device='cuda'):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)


def neighbourhood_graph(n, r):
    ii, jj = meshgrid(n, n)
    d = (ii - jj).abs()
    keep = (d >= 1) & (d <= r)
    return ii[keep], jj[keep]


def plot_shapely(shapely_list):
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots()
    for i, shapely_element in enumerate(shapely_list):
        x, y = shapely_element.exterior.xy
        ax.plot(x, y)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.grid()
    ax.set_box_aspect(1)
    plt.show()


def get_view_frustum(
        im_hw: Tuple[int, int],
        cam_intr: npt.NDArray,
        cam_pose: npt.NDArray,
        near: Optional[float] = 0.0,
        far: Optional[float] = 1.0,
) -> npt.NDArray:
    """Get corners of 3D camera view frustum of depth image
    Args:
        im_hw: Image dimensions (height, width),
        cam_intr: Camera intrinsics of the form [fx, fy, cx, cy],
        cam_pose: Camera poses of shape (4, 4),
        near: Distance of near plane
        far: Distance of far plane
    Returns:
        3D camera frustum corners: [nbl, nbr, ntr, ntl, fbl, fbr, ftr, ftl] with n/f: near/far, b/t: bottom/top,
        l/r: left/right (https://en.wikipedia.org/wiki/Viewing_frustum)
    """
    im_h, im_w = im_hw
    fx, fy, cx, cy = cam_intr
    view_frust_pts = np.array(2*[
        [-cx, -cy, 1., 1.],
        [im_w - cx, -cy, 1., 1.],
        [im_w - cx, im_h - cy, 1., 1.],
        [-cx, im_h - cy, 1., 1.],
    ])
    view_frust_pts[:, 0] /= fx
    view_frust_pts[:, 1] /= fy
    view_frust_pts[0:4, 0:3] *= near
    view_frust_pts[4:8, 0:3] *= far
    return cam_pose[None] @ view_frust_pts[..., None]


def get_frustum_intersections(
        poses: npt.NDArray,
        intrinsics: npt.NDArray,
        im_hw: List[Tuple[int, int]],
        near: Optional[float] = 5.0,
        far: Optional[float] = 100.0
) -> npt.NDArray:
    """ Get triangle (projection of camera frustum on xz-plane in cam-coords.) intersection for each pair of views.
    Args:
        poses: Camera poses (cam2ref) of shape (N, 4, 4)
        intrinsics: Camera intrinsics of shape (N, 4)
        im_hw: List of image sizes of the form [(h, w), ...]
        near: Minimum depth in m
        far: Maximum depth in m
    Returns:
        Matrix containing overlap percentage between cameras i and j of form N x N
    """
    n_poses = poses.shape[0]
    polygons = []

    # Calc 2D projections of 3D camera frustums
    for pose, intr, hw in zip(poses, intrinsics, im_hw):
        frust_pts = get_view_frustum(im_hw=hw, cam_intr=intr, cam_pose=pose, near=near, far=far)
        polygons.append(Polygon(frust_pts[[0, 1, 5, 4]][:, [0, 2], 0].tolist()))

    # Calculate intersections
    intersect_areas = np.zeros((n_poses, n_poses))
    for i, p1 in enumerate(polygons):
        p1_area = p1.area
        for j, p2 in enumerate(polygons):
            if i == j:
                intersect_areas[i, j] = 1.0
                continue
            intersect_areas[i, j] = p1.intersection(p2).area / p1_area
            # plot_shapely([p1, p2, p1.intersection(p2)])
    return intersect_areas
