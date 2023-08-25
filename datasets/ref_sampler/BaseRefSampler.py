from typing import Optional, Dict, List, Type, Union
from datasets.struct.data import Group, Sample
from datasets.BaseDataLoader import BaseDataLoader


class BaseRefSampler:
    """ Base reference view sampling class
    """
    def __init__(self, **kwargs):
        self.sample_list: List[Sample] = []

    def initialize(
            self,
            dataset: Dict[str, List[Group]],
            dataset_path: str,
            cameras: Union[List[str], List[int]],
            dataloader: Type[BaseDataLoader],
            split: Optional[Union[List[str], Dict[str, List[int]]]] = None,
    ):
        """ initializes self.sample_list
        Args:
            dataset: Cached dataset
            dataset_path: Path to dataset for loading additional data for ref.-view sampling
            cameras: List of cameras to be used
            dataloader: DataLoader implemented class for loading additional data
            split: Either list of scenes ['sceneXY', ...] or Dict of scene + sample pairs: {'sceneXY': [0, ...], ...}
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item: int) -> Sample:
        return self.sample_list[item]
