from typing import Optional
from typing_extensions import Self
import torch
import torch.nn.functional as F

import r3d3_backends


class CorrSampler(torch.autograd.Function):
    """ Correlation volume sampling layer """
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume, coords)
        ctx.radius = radius
        corr, = r3d3_backends.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = r3d3_backends.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(
            self,
            fmap1: torch.Tensor,
            fmap2: torch.Tensor,
            num_levels: Optional[int] = 4,
            radius: Optional[int] = 3
    ):
        """ Large-memory correlation-volume implementation
        Args:
            fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
            fmap2: Correlation features of shape (B, N, F, H, W)
            num_levels: Number of correlation-volume levels (Each level l has dimensions (H/2**l, W/2**l, H, W))
            radius: Volume sampling radius
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch * num * h1 * w1, 1, h2, w2)

        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch * num, h1, w1, h2 // 2 ** i, w2 // 2 ** i))
            corr = F.avg_pool2d(corr, 2, stride=2)

    def __call__(self, coords: torch.Tensor, *args) -> torch.Tensor:
        """ Sample from correlation volume
        Args:
            coords: Sampling coordinates of shape (B, N, H, W, 2)
        Returns:
            Sampled features of shape (B, N, num_levels * (radius + 1)**2, H, W)
        """
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3)
        coords = coords.contiguous().view(batch * num, 2, ht, wd)

        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords / 2 ** i, self.radius)
            out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)

    def cat(self, other: Self) -> Self:
        """ Concatenates self with given corr-volume
        Args:
            other: Other correlation-volume
        Returns:
            self
        """
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
        return self

    def __getitem__(self, index: torch.Tensor) -> Self:
        """ Index correlation volume
        Args:
            index: Mask or indices of volumes to keep
        Returns:
            Indexed instance
        """
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]
        return self

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """ Calculates all-pairs correlation
        Args:
            fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
            fmap2: Correlation features of shape (B, N, F, H, W)
        Returns:
            Correlation volume of shape (B, N, H, W, H; W)
        """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch * num, dim, ht * wd) / 4.0
        fmap2 = fmap2.reshape(batch * num, dim, ht * wd) / 4.0

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        return corr.view(batch, num, ht, wd, ht, wd)


class CorrBlockNew:
    def __init__(self, max_edges: int, num_levels: Optional[int] = 4, radius: Optional[int] = 3):
        """ Large-memory correlation-volume implementation
        Args:
            max_edges: Max. number of edges
            num_levels: Number of correlation-volume levels (Each level l has dimensions (H/2**l, W/2**l, H, W))
            radius: Volume sampling radius
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.max_edges = max_edges
        self.n_exist = 0

    def add(self, fmap1: torch.Tensor, fmap2: torch.Tensor, existing: Optional[torch.Tensor] = None) -> None:
        """ Adds correlations to block
        Args:
            fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
            fmap2: Correlation features of shape (B, N, F, H, W)
            existing: Boolean tensor of already existing edges of shape (N)
        """
        bs, n, _, h, w = fmap1.shape

        # initialize empty memory for correlation volume
        if len(self.corr_pyramid) == 0:
            for i in range(self.num_levels):
                self.corr_pyramid.append(
                    torch.zeros(bs * self.max_edges, h, w, h // 2 ** i, w // 2 ** i, dtype=fmap1.dtype, device=fmap1.device)
                )

        # Move existing correlations to the beginning of the memory
        if existing is not None:
            self.__getitem__(existing)

        mem_start, mem_end = self.n_exist, self.n_exist + bs * n
        if mem_end > self.max_edges:
            raise Exception(f'Tried to add {bs * n} edges to correrlation-volume buffer. Total number ({mem_end} edges)'
                            f' exceeds limit ({self.max_edges} edges). Try to either use the "lowmem" implementation or'
                            f' increase the maximum number of edges!')
        CorrBlockNew.corr(fmap1, fmap2, self.corr_pyramid[0][mem_start:mem_end])

        # Create pooled versions of correlation volume
        for i in range(1, self.num_levels):
            self.corr_pyramid[i][mem_start:mem_end] = F.avg_pool2d(
                self.corr_pyramid[i - 1][mem_start:mem_end].view(-1, 1, h // 2 ** (i - 1), w // 2 ** (i - 1)),
                kernel_size=2,
                stride=2
            ).view(-1, h, w, h // 2 ** i, w // 2 ** i)

        self.n_exist += bs * n

    def __call__(self, coords: torch.Tensor, *args) -> torch.Tensor:
        """ Sample from correlation volume
        Args:
            coords: Sampling coordinates of shape (B, N, H, W, 2)
        Returns:
            Sampled features of shape (B, N, num_levels * (radius + 1)**2, H, W)
        """
        out_pyramid = []
        bs, n, h, w, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3)
        coords = coords.contiguous().view(bs * n, 2, h, w)

        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i][:bs * n], coords / 2 ** i, self.radius)
            out_pyramid.append(corr.view(bs, n, -1, h, w))

        return torch.cat(out_pyramid, dim=2)

    def cat(self, other: Self) -> Self:
        """ Concatenates self with given corr-volume
        Args:
            other: Other correlation-volume
        Returns:
            self
        """
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
        return self

    def __getitem__(self, index: torch.Tensor) -> Self:
        """ Index correlation volume
        Args:
            index: Mask or indices of volumes to keep
        Returns:
            Indexed instance
        """
        if index.dtype == torch.bool:
            self.n_exist = index.sum()
        else:
            self.n_exist = len(index)
        for i in range(self.num_levels):
            self.corr_pyramid[i][:self.n_exist] = self.corr_pyramid[i][:len(index)][index]
        return self

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor, corr_volume: torch.Tensor) -> None:
        """ all-pairs correlation and save into given location
        Args:
            fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
            fmap2: Correlation features of shape (B, N, F, H, W)
            corr_volume: Tensor where correlation volume will be saved of shape (N*B, H, W, H, W)
        """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch * num, dim, ht * wd) / 4.0
        fmap2 = fmap2.reshape(batch * num, dim, ht * wd) / 4.0
        torch.matmul(fmap1.transpose(1, 2), fmap2, out=corr_volume.view(-1, ht * wd, ht * wd))


class CorrLayer(torch.autograd.Function):
    """ Low-memory correlation layer """
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = r3d3_backends.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            r3d3_backends.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None


class AltCorrBlock:
    """ Low-memory correlation volume implementation """
    @torch.cuda.amp.autocast(enabled=False)
    def __init__(self, fmaps: torch.Tensor, num_levels: Optional[int] = 4, radius: Optional[int] = 3):
        """
        Args:
            fmaps: Feature maps of shape (B, N, F, H, W)
            num_levels: Number of correlation-volume levels (Each level l has dimensions (H/2**l, W/2**l, H, W))
            radius: Volume sampling radius
        """
        self.num_levels = num_levels
        self.radius = radius

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B * N, C, H, W) / 4.0

        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H // 2 ** i, W // 2 ** i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)

    def corr_fn(self, coords: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        """ Low-memory sampling function
        Args:
            coords: Sampling coordinates of shape (B, N, H, W, S, 2)
            ii: Outgoing node for edges of shape (|E|)
            jj: Incoming node for edges of shape (|E|)
        Returns:
            Sampled features of shape (B, N, num_levels * (radius + 1)**2, H, W, S)
        """
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        for i in range(self.num_levels):
            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2 ** i).reshape(B * N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B * N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B * N,) + fmap2_i.shape[2:])

            corr = CorrLayer.apply(fmap1_i.float(), fmap2_i.float(), coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr

    def __call__(self, coords: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        """ Sample correlation volume at given coordinates
        Args:
            coords: Sampling coordinates of shape (B, N, H, W, 2)
            ii: Outgoing node for edges of shape (|E|)
            jj: Incoming node for edges of shape (|E|)
        Returns:
            Sampled features of shape (B, N, num_levels * (radius + 1)**2, H, W)
        """
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr = self.corr_fn(coords, ii, jj)

        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()
