from typing import Optional, Dict, List, Union

from datasets.struct.data import Group, Sample
from datasets.ref_sampler.BaseRefSampler import BaseRefSampler


class DefaultRefSampler(BaseRefSampler):
    """ Samples reference from past and or future views adjacent to target in temporal sequence. Removes samples
        which do not satisfy requirement (e.g. first frame in sequence if context from past is required)
        """
    def __init__(self, context: Optional[List[int]] = None, **kwargs):
        """
        Args:
            context: List of timesteps around t which should be sampled. E.g. [-1, 0, 1, 2] => t-1, t, t+1, t+2
        """
        super(DefaultRefSampler, self).__init__(**kwargs)
        if context is None:
            context = []

        self.bwd_contexts = [ctx for ctx in context if ctx < 0]
        self.fwd_contexts = [ctx for ctx in context if ctx > 0]

        self.bwd_context = 0 if len(context) == 0 else - min(0, min(context))
        self.fwd_context = 0 if len(context) == 0 else max(0, max(context))

        self.context = [v for v in range(- self.bwd_context, 0)] + \
                       [v for v in range(1, self.fwd_context + 1)]
        self.num_context = self.bwd_context + self.fwd_context

    def initialize(
            self,
            dataset: Dict[str, List[Group]],
            cameras: Union[List[str], List[int]],
            split: Optional[Union[List[str], Dict[str, List[int]]]] = None,
            *args, **kwargs):
        for scene, groups in dataset.items():
            if (split is not None) and (scene not in split):
                continue
            for i in range(len(groups)):
                if (split is not None) and (type(split) is dict) and (i not in split[scene]):
                    continue
                if all([0 <= i + cont < len(groups) for cont in [0] + self.context]):
                    sample: Sample = {
                        'sequence': scene,
                        'groups': {cont: groups[i + cont] for cont in [0] + self.context}
                    }
                    self.sample_list.append(sample)
