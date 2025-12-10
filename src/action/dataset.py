import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import Config, get_config


class BaseballActionDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', max_seq_len: int = 64, normalize: bool = True, augment: bool = False, config: Optional[Config] = None):
        self.data_root = Path(data_root)
        self.split = split
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.augment = augment
        self.config = config or get_config()
        if split == 'train':
            self.label_root = self.data_root / '1.Training' / '라벨링데이터'
        else:
            self.label_root = self.data_root / '2.Validation' / '라벨링데이터'
        print(f"[{split}] 시퀀스 수집 중...")
        self.sequences = self._collect_sequences()
        print(f"[{split}] 총 {len(self.sequences)}개 시퀀스 수집 완료")
        self.class_weights = self._compute_class_weights()

    def _collect_sequences(self) -> List[Dict]:
        sequences = []
        for action_code in self.config.CODE_TO_ACTION.keys():
            action_folder = self.label_root / f'baseball_ra_{action_code}'
            if not action_folder.exists():
                continue
            action_name = self.config.CODE_TO_ACTION[action_code]
            label = self.config.ACTION_CLASSES[action_name]
            for day_folder in action_folder.iterdir():
                if not day_folder.is_dir():
                    continue
                for seq_folder in day_folder.iterdir():
                    if not seq_folder.is_dir():
                        continue
                    json_files = sorted(seq_folder.glob('*.json'))
                    if len(json_files) > 0:
                        sequences.append({
                            'path': seq_folder,
                            'action_code': action_code,
                            'action_name': action_name,
                            'label': label,
                            'num_frames': len(json_files)
                        })
        return sequences

    def _compute_class_weights(self) -> torch.FloatTensor:
        class_counts = np.zeros(self.config.num_classes)
        for seq in self.sequences:
            class_counts[seq['label']] += 1
        total = len(self.sequences)
        weights = total / (self.config.num_classes * class_counts + 1e-6)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_info = self.sequences[idx]
        json_files = sorted(seq_info['path'].glob('*.json'))
        keypoints_list = []
        bboxes = []
        for json_path in json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            bbox = None
            points = None
            for ann in data.get('annotations', []):
                if ann.get('class') == 'person':
                    bbox = ann.get('bbox')
                elif ann.get('class') == 'player' and 'points' in ann:
                    points = ann['points']
            if points is not None:
                kp = np.array(points, dtype=np.float32).reshape(24, 3)
                keypoints_list.append(kp)
                bboxes.append(bbox if bbox else [0, 0, 1920, 1080])
        if len(keypoints_list) == 0:
            keypoints = np.zeros((self.max_seq_len, 24, 3), dtype=np.float32)
            mask = np.zeros(self.max_seq_len, dtype=np.float32)
        else:
            keypoints = np.stack(keypoints_list, axis=0)
            if self.normalize:
                keypoints = self._normalize_sequence(keypoints, bboxes)
            keypoints, mask = self._adjust_sequence_length(keypoints)
            if self.augment and self.split == 'train':
                keypoints = self._augment(keypoints, mask)
        keypoints_flat = keypoints.reshape(self.max_seq_len, -1)
        return {
            'keypoints': torch.FloatTensor(keypoints_flat),
            'label': torch.LongTensor([seq_info['label']]).squeeze(),
            'mask': torch.FloatTensor(mask),
            'seq_len': min(seq_info['num_frames'], self.max_seq_len),
            'action_name': seq_info['action_name']
        }

    def _normalize_sequence(self, keypoints: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        normalized = keypoints.copy()
        for t in range(len(keypoints)):
            bbox = bboxes[t]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            normalized[t, :, 0] = (keypoints[t, :, 0] - cx) / w
            normalized[t, :, 1] = (keypoints[t, :, 1] - cy) / h
        return normalized

    def _adjust_sequence_length(self, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(keypoints)
        if T < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - T, 24, 3), dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)
            mask = np.array([1.0] * T + [0.0] * (self.max_seq_len - T), dtype=np.float32)
        else:
            indices = np.linspace(0, T - 1, self.max_seq_len, dtype=int)
            keypoints = keypoints[indices]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        return keypoints, mask

    def _augment(self, keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            keypoints = self._horizontal_flip(keypoints)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, keypoints.shape).astype(np.float32)
            noise_mask = mask[:, None, None].astype(np.float32)
            keypoints[:, :, :2] += noise[:, :, :2] * noise_mask[:, :, :2]
        if np.random.rand() < 0.2:
            drop_prob = 0.1
            drop_mask = (np.random.rand(self.max_seq_len) > drop_prob).astype(np.float32)
            drop_mask = drop_mask * mask
            keypoints = keypoints * drop_mask[:, None, None]
        return keypoints

    def _horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        flipped = keypoints.copy()
        flipped[:, :, 0] = -flipped[:, :, 0]
        for i, j in self.config.FLIP_PAIRS:
            flipped[:, [i, j]] = flipped[:, [j, i]]
        return flipped

    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {}
        for seq in self.sequences:
            name = seq['action_name']
            distribution[name] = distribution.get(name, 0) + 1
        return distribution


def create_dataloaders(data_root: str, batch_size: int = 32, max_seq_len: int = 64, num_workers: int = 4, use_weighted_sampler: bool = True, config: Optional[Config] = None) -> Tuple[DataLoader, DataLoader, torch.FloatTensor]:
    config = config or get_config()
    train_dataset = BaseballActionDataset(data_root=data_root, split='train', max_seq_len=max_seq_len, normalize=True, augment=True, config=config)
    val_dataset = BaseballActionDataset(data_root=data_root, split='val', max_seq_len=max_seq_len, normalize=True, augment=False, config=config)
    if use_weighted_sampler:
        sample_weights = []
        for seq in train_dataset.sequences:
            sample_weights.append(train_dataset.class_weights[seq['label']].item())
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, train_dataset.class_weights


if __name__ == '__main__':
    config = get_config()
    print("데이터셋 테스트...")
    dataset = BaseballActionDataset(data_root=config.data_root, split='train', max_seq_len=64, normalize=True, augment=False)
    print(f"\n총 시퀀스 수: {len(dataset)}")
    print(f"\n클래스 분포:")
    for name, count in sorted(dataset.get_class_distribution().items()):
        print(f"  {name}: {count}")
    print(f"\n클래스 가중치:")
    for i, w in enumerate(dataset.class_weights):
        print(f"  {i}: {w:.4f}")
    print("\n샘플 데이터 로드 테스트...")
    sample = dataset[0]
    print(f"  keypoints shape: {sample['keypoints'].shape}")
    print(f"  label: {sample['label']}")
    print(f"  mask sum: {sample['mask'].sum()}")
    print(f"  action_name: {sample['action_name']}")
