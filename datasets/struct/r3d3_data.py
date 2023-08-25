from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict, NotRequired


class R3D3Data(TypedDict):
    """
    pose: R3D3 generated pose of shape [7]
    maps: Path to R3D3 maps npz file containing the following arrays
        'disp' - Geometrically inferred disparity of shape [H/8, W/8]
        'disp_up' - Upsampled geometrically inferred disparity of shape [H, W]
        'conf' - Depth confidence of shape [H/8, W/8] with values in range [0, 1]
    """
    pose: NotRequired[npt.NDArray[np.float] | None]
    maps: NotRequired[str | None]

