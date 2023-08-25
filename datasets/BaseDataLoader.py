from abc import ABC
from typing import Optional, Dict, List

import numpy as np
import numpy.typing as npt
from PIL.Image import Image

from vidar.utils.decorators import iterate1


class BaseDataLoader(ABC):
    """ This abstract class defines the interface for loading data. Each dataset has to inherit from BaseDataLoader
        and implement respective datafields which are used (according to provided labels).
    """
    @staticmethod
    @iterate1
    def get_image(filename: str) -> Image:
        """
        Args:
            filename: Path to file
        Returns:
            image: PIL Image
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_depth(filename: str) -> npt.NDArray[np.float32]:
        """
        Args:
            filename: Path to file
        Returns:
            depth_map: Depth map in [m] (H, W)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_points(filename: str) -> npt.NDArray[np.float32]:
        """
        Args:
            filename: Path to file
        Returns:
            points: Point-cloud (N, 3)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_mask(filename: str) -> npt.NDArray[np.int32]:
        """
        Args:
            filename: Path to file
        Returns:
            mask: Mask (H, W)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_instance(filename: str) -> npt.NDArray[np.int32]:
        """
        Args:
            filename: Path to file
        Returns:
            instance_map: instance map where val = id, -1 = background (H, W)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_semantic(filename: str) -> npt.NDArray[np.int32]:  # ToDo: Standardize semantic ids in metadata
        """
        Args:
            filename: Path to file
        Returns:
            instance_map: semantic map where val = semantic_id (H, W)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_optical_flow(filename: str) -> Optional[npt.NDArray[np.float32]]:
        """
        Args:
            filename: Path to file
        Returns:
            optical_flow: (H, W, 2) or None if not available
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_scene_flow(filename: str) -> Optional[npt.NDArray[np.float32]]:
        """
        Args:
            filename: Path to file
        Returns:
            scene_flow: (H, W, 3) or None if not available
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_optical_flow_mask(filename: str) -> Optional[npt.NDArray[np.int32]]:
        """
        Args:
            filename: Path to file
        Returns:
            opt_flow_mask: Mask where optical flow is valid (H, W)
        """
        raise NotImplementedError

    @staticmethod
    @iterate1
    def get_others(root: str, others: Dict, labels: List[str]) -> Dict[str, npt.NDArray]:
        """
        Args:
            root: Root of dataset in case files have to be loaded
            others: Full content of others field
            labels: Full list of labels which should be loaded
        Returns:
            others: Dict of others data
        """
        raise NotImplementedError
