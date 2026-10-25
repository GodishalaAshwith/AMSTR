"""
Unit Test for MixedScriptRatioDataSet & MixedScriptRatioSampler
Verifies:
  1. Ratio bucketing: every sample in a batch has the exact same image width/height and ratio.
  2. Batch composition: strictly 60% Latin (38) and 40% Indic (26).
  3. PyTorch default_collate executes cleanly without "Trying to resize storage" error.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.data.mixed_script_dataset import MixedScriptRatioSampler


class MockSubDataset(Dataset):
    def __init__(self, num_samples, script_id):
        self.num_samples = num_samples
        self.script_id = script_id
        # Samples spread across ratios 1 to 4
        self.wh_ratio = np.random.randint(1, 5, size=num_samples)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ds_width = True
        self.seed = 42
        self.need_reset = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, properties):
        w, h, idx, ratio = properties[:4]
        # Image has exact spatial dimensions (h, w)
        img = torch.randn(3, h, w)
        label = torch.randint(1, 50, (25,))
        return [img, label]


class MockMixedDataset(Dataset):
    def __init__(self, latin_len=300, indic_len=200):
        self.latin_ds = MockSubDataset(latin_len, script_id=0)
        self.indic_ds = MockSubDataset(indic_len, script_id=1)
        self.ds_width = True
        self.wh_ratio = self.latin_ds.wh_ratio
        self.wh_ratio_sort = self.latin_ds.wh_ratio_sort
        self.need_reset = False

    def __len__(self):
        return len(self.latin_ds) + len(self.indic_ds)

    def __getitem__(self, properties):
        w, h, idx, ratio = properties[:4]
        script_type = properties[4] if len(properties) >= 5 else 0
        target = self.latin_ds if script_type == 0 else self.indic_ds
        return target.__getitem__([w, h, idx, ratio])


def run_tests():
    print("=" * 60)
    print("Testing MixedScriptRatioSampler with Ratio Bucketing")
    print("=" * 60)

    dataset = MockMixedDataset(latin_len=400, indic_len=200)
    batch_size = 64
    sampler = MixedScriptRatioSampler(
        data_source=dataset,
        scales=[[128, 32]],
        first_bs=batch_size,
        latin_ratio=0.60,
        is_training=True,
    )

    batches = list(sampler)
    print(f"Generated {len(batches)} ratio-bucketed batches.")
    assert len(batches) > 0, "No batches generated!"

    expected_latin = 38
    expected_indic = 26

    for b_idx, batch in enumerate(batches[:5]):
        assert len(batch) == batch_size, f"Batch size mismatch: {len(batch)} vs {batch_size}"
        
        # Verify uniform ratio and width across the entire batch
        ratios_in_batch = {item[3] for item in batch}
        widths_in_batch = {item[0] for item in batch}
        assert len(ratios_in_batch) == 1, f"Multiple ratios in batch {b_idx}: {ratios_in_batch}"
        assert len(widths_in_batch) == 1, f"Multiple widths in batch {b_idx}: {widths_in_batch}"

        latin_count = sum(1 for item in batch if item[4] == 0)
        indic_count = sum(1 for item in batch if item[4] == 1)
        assert latin_count == expected_latin, f"Latin count mismatch: {latin_count} vs {expected_latin}"
        assert indic_count == expected_indic, f"Indic count mismatch: {indic_count} vs {expected_indic}"

        # Fetch actual samples from dataset
        samples = [dataset[item] for item in batch]
        
        # Test PyTorch default_collate (this was failing in Colab!)
        collated_batch = default_collate(samples)
        img_batch, label_batch = collated_batch
        assert img_batch.shape[0] == batch_size
        assert img_batch.shape[1] == 3
        assert img_batch.shape[2] == 32
        print(f"  -> Batch {b_idx}: ratio={list(ratios_in_batch)[0]}, img tensor shape={list(img_batch.shape)} [PASSED]")

    print("=" * 60)
    print("ALL TESTS PASSED! PyTorch Collation is 100% Valid and Uniform.")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
