# EAST Train/Val/Test Split Strategy

## Overview

This directory contains train/val/test splits for EAST dataset evaluation, aligned with Paper131 protocol.

## Official Data Discovery

**Total EAST Shots**: 1681 (398 disruptive + 1283 non-disruptive)
- **Source**: `artifacts/east_mainA12_paper131_feature_contract_a12_20260517a12c`
- **Shot ID Range**: 54098 to 96905
- **Disruption Ratio**: 23.7%

**Paper131 Protocol**: `east_prequential_active_online_v1`
- **D_stream (test)**: 1571 shots (388D + 1183ND) - Official chronological streaming evaluation set
- **D_train/D_val pool**: 110 shots (10D + 100ND) - Reserved for fine-tuning and threshold selection
- **Key property**: D_stream and train/val pools are **disjoint but temporally interleaved** (not sequential)

## Split Files

### 1. `east_paper131_split_all_in.json` ✓ RECOMMENDED

**Configuration**:
- **Train**: 110 shots (10D + 100ND) - All available non-stream shots
- **Val**: 0 shots - Use train set for threshold selection
- **Test**: 1571 shots (388D + 1183ND) - Official D_stream

**Usage**:
```python
import json

with open('config/splits/east_paper131_split_all_in.json') as f:
    split = json.load(f)

train_shots = split['train']['disruption_shots'] + split['train']['non_disruption_shots']
test_shots = split['test']['disruption_shots'] + split['test']['non_disruption_shots']

print(f"Train: {len(train_shots)} shots (10D + 100ND)")
print(f"Test: {len(test_shots)} shots (388D + 1183ND)")
```

**Rationale**:
- ✓ Matches Paper131 specification exactly (10D + 100ND for fine-tuning)
- ✓ Maximizes training data for S-CORAL domain adaptation
- ✓ Standard practice: use train set for threshold selection in online learning
- ✓ Test on official D_stream for final metrics

### 2. `east_paper131_split_with_val.json` (Alternative)

**Configuration**:
- **Train**: 88 shots (8D + 80ND) - 80% of available shots
- **Val**: 22 shots (2D + 20ND) - 20% of available shots
- **Test**: 1571 shots (388D + 1183ND) - Official D_stream

**Rationale**:
- Separate validation set for hyperparameter tuning
- More conservative approach with held-out validation
- Use if you need independent threshold/calibration validation

## Important Notes

### Temporal Ordering

⚠️ **Train and test shots are temporally INTERLEAVED, not sequential**:
- Train range: 54314 to 96845
- Test range: 54098 to 96905
- This is **by design** in Paper131 protocol (predeclared split, not temporal split)

**No Data Leakage**:
- ✓ Shot sets are completely disjoint (no overlap)
- ✓ Split is predeclared (not adaptive to test set)
- ✓ Follows standard prequential online learning protocol

### Paper131 Alignment

The official Paper131 paper claims "595 disruptive + 1781 non-disruptive = 2376 total" but actual clean data after conflict resolution is **1681 shots**. The 2376 figure likely represents pre-cleaned raw data.

**Our splits use the actual available clean data**: 398D + 1283ND = 1681 shots

### Checkpoint Compatibility

When using pre-trained checkpoints, verify they were trained on:
1. **Upstream data** (JTEXT or other sources) - OK to use
2. **EAST D_train** (predeclared subset) - OK to use
3. **NOT on EAST D_stream** (our test set) - Would invalidate evaluation

Check checkpoint `training_config.json` for data sources.

## Verification Checklist

✓ No shot ID overlap between train and test
✓ Total shots = 1681 (398D + 1283ND)
✓ Test set = official D_stream from Paper131 artifact
✓ Train set = exactly 10D + 100ND (Paper131 spec)
✓ All shot IDs verified against HDF5 availability

## Usage Examples

### Load Split for Training

```python
import json

def load_east_split(split_file='config/splits/east_paper131_split_all_in.json'):
    with open(split_file) as f:
        split = json.load(f)
    
    return {
        'train_disruption': split['train']['disruption_shots'],
        'train_non_disruption': split['train']['non_disruption_shots'],
        'test_disruption': split['test']['disruption_shots'],
        'test_non_disruption': split['test']['non_disruption_shots']
    }

split = load_east_split()
print(f"Train: {len(split['train_disruption']) + len(split['train_non_disruption'])} shots")
print(f"Test: {len(split['test_disruption']) + len(split['test_non_disruption'])} shots")
```

### Stratified Batch Sampling

```python
import random

def sample_stratified_batch(split, batch_size=16, disruption_ratio=0.237):
    """Sample batch maintaining disruption ratio"""
    n_disruption = int(batch_size * disruption_ratio)
    n_non_disruption = batch_size - n_disruption
    
    batch_d = random.sample(split['train_disruption'], n_disruption)
    batch_nd = random.sample(split['train_non_disruption'], n_non_disruption)
    
    return batch_d + batch_nd
```

### Filter HDF5 Data by Split

```python
import h5py

def load_shot_data(shot_id, hdf5_dir='/data/EAST/unified_hdf5'):
    """Load features for a shot"""
    h5_path = f"{hdf5_dir}/shot_{shot_id}.h5"
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        labels = f['labels'][:]
    return features, labels

# Load all train shots
split = load_east_split()
train_data = []
for shot_id in split['train_disruption'] + split['train_non_disruption']:
    features, labels = load_shot_data(shot_id)
    train_data.append((shot_id, features, labels))
```

## Citation

If using these splits, cite the Paper131 artifact:
```
Source: artifacts/east_mainA12_paper131_feature_contract_a12_20260517a12c
Protocol: east_prequential_active_online_v1
Split: D_stream (test) + D_train/D_val (train) pools
```

## Changelog

- **2026-06-14**: Initial split creation aligned with Paper131 protocol
  - Generated from official clean shot lists and chronological_stream_split.json
  - Verified all 1681 shots have HDF5 backing
  - Created two split variants (all-in and with-val)
