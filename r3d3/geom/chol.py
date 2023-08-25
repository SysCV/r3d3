import torch


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        return_chol = False
        # don't crash training if cholesky decomp fails
        U = None
        U_inv = None
        H_inv = None
        try:
            U = torch.linalg.cholesky(H)
            if not return_chol:
                xs = torch.cholesky_solve(b, U)
                ctx.save_for_backward(U, xs)
                ctx.failed = False
            else:
                identity = torch.eye(U.shape[1], device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1)
                U_inv = torch.triangular_solve(identity, U, upper=False)[0]
                U_conj_inv = torch.triangular_solve(identity, torch.transpose(U, -1, -2), upper=True)[0]
                H_inv = U_conj_inv @ U_inv
                xs = H_inv @ b

            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs, U, U_inv, H_inv

    @staticmethod
    def backward(ctx, grad_x, U, U_inv, H_inv):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2))

        return dH, dz


def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm * H) * I

    H = H.permute(0, 1, 3, 2, 4)
    H = H.reshape(B, N * D, N * D)
    b = b.reshape(B, N * D, 1)

    x, L, L_inv, H_inv = CholeskySolver.apply(H, b)
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """

    B, P, M, D, HW = E.shape
    H = H.permute(0, 1, 3, 2, 4).reshape(B, P * D, P * D)
    E = E.permute(0, 1, 3, 2, 4).reshape(B, P * D, M * HW)
    Q = (1.0 / C).view(B, M * HW, 1)

    # damping
    I = torch.eye(P * D).to(H.device)
    H = H + (ep + lm * H) * I

    v = v.reshape(B, P * D, 1)
    w = w.reshape(B, M * HW, 1)

    Et = E.transpose(1, 2)
    S = H - torch.matmul(E, Q * Et)
    v = v - torch.matmul(E, Q * w)

    dx, L, L_inv, S_inv = CholeskySolver.apply(S, v)

    x_cov, z_cov = None, None
    if L_inv is not None:
        F = torch.matmul(Q * Et, L_inv)  # K*HW x D*P
        F2 = torch.pow(F, 2)
        z_cov = Q.squeeze(-1) + F2.sum(dim=-1)
        z_cov = z_cov.view(B, M, HW)
        x_cov = S_inv.view(B, P, D, P, D)

    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    return dx, dz, x_cov, z_cov
