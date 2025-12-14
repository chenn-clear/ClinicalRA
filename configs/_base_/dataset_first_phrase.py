data_prefix = ''
train_pipeline = [
    dict(type='Pipeline_AT'),
    dict(type='ToTensor', keys=['gt_label', 'attributes']),
    dict(type='Collect', keys=['img', 'gt_label', 'attributes'])
]

test_pipeline = [
    dict(type='Pipeline_AT', test_mode=True),
    dict(type='ToTensor', keys=['attributes']),
    dict(type='Collect', keys=['img', 'attributes'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=12,
    train=dict(
        type='AT',
        data_prefix=data_prefix,
        ann_file='',
        pipeline=train_pipeline),
    val=dict(
        type='AT',
        data_prefix=data_prefix,
        ann_file='',
        pipeline=test_pipeline),
    test=dict(
        type='AT',
        data_prefix=data_prefix, 
        ann_file='',
        pipeline=test_pipeline))