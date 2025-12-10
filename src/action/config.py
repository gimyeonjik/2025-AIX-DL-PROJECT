import os
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class Config:
    data_root: str = os.environ.get('BASEBALL_DATA_ROOT', './data')
    save_dir: str = os.environ.get('BASEBALL_SAVE_DIR', './checkpoints')
    max_seq_len: int = 64
    num_keypoints: int = 24
    input_dim: int = 72
    num_classes: int = 13
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    early_stopping_patience: int = 15
    label_smoothing: float = 0.1
    augment: bool = True
    use_weighted_sampler: bool = True
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42

    ACTION_CLASSES: Dict[str, int] = field(default_factory=lambda: {
        'catch_throw': 0,
        'foul_fly': 1,
        'fly_ground': 2,
        'foul_tip': 3,
        'hit_by_pitch': 4,
        'home_run': 5,
        'swing': 6,
        'passed_ball': 7,
        'pick_off': 8,
        'pitch': 9,
        'pitch_setup': 10,
        'play_up': 11,
        'runner_run': 12
    })

    CODE_TO_ACTION: Dict[str, str] = field(default_factory=lambda: {
        'ct': 'catch_throw',
        'ff': 'foul_fly',
        'fg': 'fly_ground',
        'ft': 'foul_tip',
        'hb': 'hit_by_pitch',
        'hh': 'home_run',
        'hs': 'swing',
        'pb': 'passed_ball',
        'po': 'pick_off',
        'pp': 'pitch',
        'ps': 'pitch_setup',
        'pu': 'play_up',
        'rr': 'runner_run'
    })

    CLASS_NAMES_KR: Dict[int, str] = field(default_factory=lambda: {
        0: '포수 송구',
        1: '파울 플라이',
        2: '플라이/땅볼 아웃',
        3: '파울 팁',
        4: '몸에 맞는 공',
        5: '홈런 타격',
        6: '스윙',
        7: '패스트볼',
        8: '견제',
        9: '투구',
        10: '투구 준비',
        11: '플레이 업',
        12: '주자 달리기'
    })

    FLIP_PAIRS: List[List[int]] = field(default_factory=lambda: [
        [1, 2],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
    ])


def get_config() -> Config:
    return Config()
