import torch
from pytorch3d.transforms import matrix_to_quaternion


def pose_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """ Converts poses in matrix [..., 4, 4] format into translation + quaternion format [..., 7]
    Args:
        Pose matrix of shape [..., 4, 4]
    Returns:
        quat: Pose with [x, y, z, quaternions]
    """
    quat = torch.cat((
        matrix[..., 0:3, 3],
        matrix_to_quaternion(matrix[..., 0:3, 0:3])
    ), dim=-1)[..., [0, 1, 2, 4, 5, 6, 3]]
    return quat