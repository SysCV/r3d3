import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = 0.2


def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)


def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht, device=disps.device, dtype=torch.float32),
        torch.arange(wd, device=disps.device, dtype=torch.float32))

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[..., -1] = 1.0
        return pts, J

    return pts, None


def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D * d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
            fx * d, o, -fx * X * d * d, o,
            o, fy * d, -fy * Y * d * d, o,
            # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:, :, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d, o, o, o, Z, -Y,
                o, d, o, -Z, o, X,
                o, o, d, Y, -X, o,
                o, o, o, o, o, o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d, o, o, o, Z, -Y, X,
                o, d, o, -Z, o, X, Y,
                o, o, d, Y, -X, o, Z,
                o, o, o, o, o, o, o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None


def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)

        Jz = Gij[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def projective_transform_rel(poses, rel_poses, depths, intrinsics, ii, jj, cii, cjj, jacobian=False,
                             return_depth=False):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:, cii], intrinsics[:, cii], jacobian=jacobian)

    # transform
    Gcci = rel_poses[:, cii]  # (t_i, c0) -> (t_i, c)
    Gccj = rel_poses[:, cjj]  # (t_j, c0) -> (t_j, c)
    Gicj = Gccj * poses[:, jj] * poses[:, ii].inv()  # (t_i, c0) -> (t_j, c)
    Gcicj = Gicj * Gcci.inv()  # (t_i, c) -> (t_j, c)

    X1, Ja = actp(Gcicj, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, cjj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        Jx = torch.matmul(Jp, Ja)
        Jj = Gccj[:, :, None, None, None].adjT(Jx)
        Ji = -Gicj[:, :, None, None, None].adjT(Jx)

        Jz = Gcicj[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht, device=disps.device, dtype=torch.float32),
        torch.arange(wd, device=disps.device, dtype=torch.float32))

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[..., :2] - coords0, valid


def _bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def bilinear_sampler(img, coords):
    """ Wrapper for bilinear sampler for inputs with extra batch dimensions """
    unflatten = False
    if len(img.shape) == 5:
        unflatten = True
        b, n, c, h, w = img.shape
        img = img.view(b * n, c, h, w)
        coords = coords.view(b * n, h, w, 2)

    img1 = _bilinear_sampler(img, coords)

    if unflatten:
        return img1.view(b, n, c, h, w)

    return img1
