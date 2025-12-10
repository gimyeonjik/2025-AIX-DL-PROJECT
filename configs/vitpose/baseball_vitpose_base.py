_base_ = ['mmpose::_base_/default_runtime.py']

auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')

codec = dict(
    heatmap_size=(48, 64),
    input_size=(192, 256),
    sigma=2,
    type='UDPHeatmap'
)

custom_imports = dict(
    allow_failed_imports=False,
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper']
)

custom_hooks = [dict(type='SyncBuffersHook')]

default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'
    ),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        rule='greater',
        save_best='baseball/AP',
        type='CheckpointHook'
    ),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook')
)

default_scope = 'mmpose'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

load_from = 'pretrained/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'

log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=None
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=24,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.1
    ),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys=dict(
            bias=dict(decay_multi=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)
        )
    )
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210, milestones=[170, 200], gamma=0.1, by_epoch=True)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict()

data_mode = 'topdown'
data_root = 'data/baseball_pose/'

dataset_info = dict(
    dataset_name='baseball_pose',
    keypoint_info={
        0: dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='eye_right', id=1, color=[51, 153, 255], type='upper', swap='eye_left'),
        2: dict(name='eye_left', id=2, color=[51, 153, 255], type='upper', swap='eye_right'),
        3: dict(name='neck', id=3, color=[51, 153, 255], type='upper', swap=''),
        4: dict(name='chest', id=4, color=[51, 153, 255], type='upper', swap=''),
        5: dict(name='right_shoulder', id=5, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        6: dict(name='left_shoulder', id=6, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        7: dict(name='right_elbow', id=7, color=[255, 128, 0], type='upper', swap='left_elbow'),
        8: dict(name='left_elbow', id=8, color=[0, 255, 0], type='upper', swap='right_elbow'),
        9: dict(name='right_wrist', id=9, color=[255, 128, 0], type='upper', swap='left_wrist'),
        10: dict(name='left_wrist', id=10, color=[0, 255, 0], type='upper', swap='right_wrist'),
        11: dict(name='right_fingertips', id=11, color=[255, 128, 0], type='upper', swap='left_fingertips'),
        12: dict(name='left_fingertips', id=12, color=[0, 255, 0], type='upper', swap='right_fingertips'),
        13: dict(name='waist', id=13, color=[51, 153, 255], type='lower', swap=''),
        14: dict(name='right_hip', id=14, color=[255, 128, 0], type='lower', swap='left_hip'),
        15: dict(name='left_hip', id=15, color=[0, 255, 0], type='lower', swap='right_hip'),
        16: dict(name='right_knee', id=16, color=[255, 128, 0], type='lower', swap='left_knee'),
        17: dict(name='left_knee', id=17, color=[0, 255, 0], type='lower', swap='right_knee'),
        18: dict(name='right_ankle', id=18, color=[255, 128, 0], type='lower', swap='left_ankle'),
        19: dict(name='left_ankle', id=19, color=[0, 255, 0], type='lower', swap='right_ankle'),
        20: dict(name='right_tiptoe', id=20, color=[255, 128, 0], type='lower', swap='left_tiptoe'),
        21: dict(name='left_tiptoe', id=21, color=[0, 255, 0], type='lower', swap='right_tiptoe'),
        22: dict(name='right_heel', id=22, color=[255, 128, 0], type='lower', swap='left_heel'),
        23: dict(name='left_heel', id=23, color=[0, 255, 0], type='lower', swap='right_heel'),
    },
    skeleton_info={
        0: dict(link=('head', 'neck'), id=0, color=[51, 153, 255]),
        1: dict(link=('eye_right', 'eye_left'), id=1, color=[51, 153, 255]),
        2: dict(link=('head', 'eye_right'), id=2, color=[51, 153, 255]),
        3: dict(link=('head', 'eye_left'), id=3, color=[51, 153, 255]),
        4: dict(link=('neck', 'chest'), id=4, color=[51, 153, 255]),
        5: dict(link=('chest', 'right_shoulder'), id=5, color=[255, 128, 0]),
        6: dict(link=('chest', 'left_shoulder'), id=6, color=[0, 255, 0]),
        7: dict(link=('right_shoulder', 'right_elbow'), id=7, color=[255, 128, 0]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_elbow', 'right_wrist'), id=9, color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('right_wrist', 'right_fingertips'), id=11, color=[255, 128, 0]),
        12: dict(link=('left_wrist', 'left_fingertips'), id=12, color=[0, 255, 0]),
        13: dict(link=('chest', 'waist'), id=13, color=[51, 153, 255]),
        14: dict(link=('waist', 'right_hip'), id=14, color=[255, 128, 0]),
        15: dict(link=('waist', 'left_hip'), id=15, color=[0, 255, 0]),
        16: dict(link=('right_hip', 'right_knee'), id=16, color=[255, 128, 0]),
        17: dict(link=('left_hip', 'left_knee'), id=17, color=[0, 255, 0]),
        18: dict(link=('right_knee', 'right_ankle'), id=18, color=[255, 128, 0]),
        19: dict(link=('left_knee', 'left_ankle'), id=19, color=[0, 255, 0]),
        20: dict(link=('right_ankle', 'right_tiptoe'), id=20, color=[255, 128, 0]),
        21: dict(link=('left_ankle', 'left_tiptoe'), id=21, color=[0, 255, 0]),
        22: dict(link=('right_ankle', 'right_heel'), id=22, color=[255, 128, 0]),
        23: dict(link=('left_ankle', 'left_heel'), id=23, color=[0, 255, 0]),
    },
    joint_weights=[1.0] * 24,
    sigmas=[0.05] * 24,
    flip_pairs=[
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
    ]
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        metainfo=dataset_info
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=val_pipeline,
        test_mode=True,
        metainfo=dataset_info
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val.json',
    use_area=True,
    iou_type='keypoints',
    prefix='baseball'
)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends
)
