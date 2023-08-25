from typing import Optional, Dict, List, Union
from datasets.struct.data import Group, Sample

from datasets.ref_sampler.DefaultRefSampler import DefaultRefSampler


class R3D3RefSampler(DefaultRefSampler):
    """ DefaultRefSampler but removes samples which do not have an R3D3 prediction (pose + depth + conf) """
    def __init__(self, context: Optional[List[int]] = None, filter_context: Optional[bool] = True, **kwargs):
        """
        Args:
            context: List of timesteps around t which should be sampled. E.g. [-2, 0, +3] => t-2, t, t+3
            filter_context: True - filters if R3D3 neither present in target nor in ref.-view, False - only former
        """
        super(R3D3RefSampler, self).__init__(context, **kwargs)
        self.filter_context = filter_context

    def initialize(
            self,
            dataset: Dict[str, List[Group]],
            cameras: Union[List[str], List[int]],
            split: Optional[Union[List[str], Dict[str, List[int]]]] = None,
            *args, **kwargs):
        super().initialize(dataset, split, *args, **kwargs)
        sample_list_new = []
        for sample in self.sample_list:
            add_sample = True
            for idx, group in sample['groups'].items():
                frames = [group['frames'][cam] for cam in cameras]
                if not all([('maps' in f['others'] and f['others']['maps'] is not None) for f in frames]):
                    if (idx == 0) or self.filter_context:
                        add_sample = False
            if add_sample:
                sample_list_new.append(sample)
        self.sample_list = sample_list_new
