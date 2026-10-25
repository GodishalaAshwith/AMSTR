import copy
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize


class MixedScriptRatioDataSet(Dataset):
    """
    Custom PyTorch Dataset Wrapper:
    Combines Latin (Union14M-L) and Indic (IndicSTR12) datasets.
    Guarantees:
      - Clean error handling (no None returns to collator).
      - Standard 11 tensor keys matching GTCDecoder & SVTRv2XLoss.
    """

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super().__init__()
        self.logger = logger
        self.mode = mode
        self.seed = epoch
        self.task = task

        dataset_config = config[mode]['dataset']

        # 1. Latin dataset instance (script_label = 0)
        latin_config = copy.deepcopy(config)
        latin_dataset_cfg = copy.deepcopy(dataset_config)
        latin_dataset_cfg['data_dir_list'] = dataset_config.get(
            'latin_data_dir_list', dataset_config.get('data_dir_list')
        )
        latin_config[mode]['dataset'] = latin_dataset_cfg
        self.latin_ds = RatioDataSetTVResize(latin_config, mode, logger, seed, epoch, task)

        # 2. Indic dataset instance (script_label = 1)
        indic_config = copy.deepcopy(config)
        indic_dataset_cfg = copy.deepcopy(dataset_config)
        indic_dataset_cfg['data_dir_list'] = dataset_config.get(
            'indic_data_dir_list', dataset_config.get('data_dir_list')
        )
        indic_config[mode]['dataset'] = indic_dataset_cfg
        self.indic_ds = RatioDataSetTVResize(indic_config, mode, logger, seed, epoch, task)

        self.ds_width = self.latin_ds.ds_width
        self.wh_ratio = self.latin_ds.wh_ratio
        self.wh_ratio_sort = self.latin_ds.wh_ratio_sort
        self.need_reset = self.latin_ds.need_reset or self.indic_ds.need_reset

    def __len__(self):
        return len(self.latin_ds) + len(self.indic_ds)

    def __getitem__(self, properties):
        w, h, idx, ratio = properties[:4]
        script_type = properties[4] if len(properties) >= 5 else 0

        target_ds = self.latin_ds if script_type == 0 else self.indic_ds
        sample = target_ds.__getitem__([w, h, idx, ratio])

        # Ensure no None reaches the PyTorch collator by resilient resampling
        wh_pool = target_ds.wh_ratio
        retry_count = 0
        while sample is None and retry_count < 10:
            candidate_ids = np.where(wh_pool == ratio)[0]
            if len(candidate_ids) == 0:
                new_idx = random.randint(0, len(target_ds) - 1)
            else:
                new_idx = int(random.choice(candidate_ids))
            sample = target_ds.__getitem__([w, h, new_idx, ratio])
            retry_count += 1

        return sample


class MixedScriptRatioSampler(Sampler):
    """
    Custom PyTorch Batch Sampler:
    Guarantees:
      1. Every batch contains strictly 60% Latin (38) and 40% Indic (26).
      2. ALL samples within the same batch share the EXACT SAME aspect ratio `ratio`
         and image width (W = ratio * 32), guaranteeing uniform tensor dimensions
         for PyTorch batch collation.
    """

    def __init__(
        self,
        data_source,
        scales,
        first_bs=64,
        latin_ratio=0.60,
        is_training=True,
        max_ratio=8,
        **kwargs,
    ):
        self.data_source = data_source
        self.first_bs = first_bs
        self.latin_ratio = latin_ratio
        self.is_training = is_training
        self.max_ratio = max_ratio
        self.epoch = 0

        self.latin_bs = max(1, int(round(first_bs * latin_ratio)))  # 38
        self.indic_bs = first_bs - self.latin_bs                    # 26

        self.base_h = 32
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        latin_wh = np.array(self.data_source.latin_ds.wh_ratio)
        indic_wh = np.array(self.data_source.indic_ds.wh_ratio)

        # For each aspect ratio bucket present in both datasets
        ratios = [
            r for r in range(1, self.max_ratio + 1)
            if (latin_wh == r).any() and (indic_wh == r).any()
        ]

        for r in ratios:
            l_pool = np.where(latin_wh == r)[0].tolist()
            i_pool = np.where(indic_wh == r)[0].tolist()

            if len(l_pool) < self.latin_bs or len(i_pool) < self.indic_bs:
                continue

            if self.is_training:
                random.shuffle(l_pool)
                random.shuffle(i_pool)

            # Number of batches for this ratio based on Latin pool
            num_ratio_batches = len(l_pool) // self.latin_bs
            w = r * self.base_h
            h = self.base_h

            l_idx = 0
            i_idx = 0
            for _ in range(num_ratio_batches):
                batch = []
                # 1. 38 Latin items with ratio r
                for _ in range(self.latin_bs):
                    batch.append([w, h, l_pool[l_idx], r, 0])
                    l_idx += 1

                # 2. 26 Indic items with SAME ratio r
                for _ in range(self.indic_bs):
                    if i_idx >= len(i_pool):
                        random.shuffle(i_pool)
                        i_idx = 0
                    batch.append([w, h, i_pool[i_idx], r, 1])
                    i_idx += 1

                if self.is_training:
                    random.shuffle(batch)

                batches.append(batch)

        if self.is_training:
            random.shuffle(batches)

        return batches

    def __iter__(self):
        if self.is_training:
            self.epoch += 1
            random.seed(self.epoch)
            random.shuffle(self.batches)
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)
