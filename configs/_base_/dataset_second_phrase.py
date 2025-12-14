data_prefix = ''
train_pipeline = [
    dict(type='Pipeline_RAT'),
    dict(type='ToTensor', keys=['gt_label', 'attributes', 'retrieved_attributes']),
    dict(type='Collect', keys=['img', 'gt_label', 'attributes','retrieved_images', 'retrieved_attributes', 'retrieved_labels'])
]

test_pipeline = [
    dict(type='Pipeline_RAT', test_mode=True),
    dict(type='ToTensor', keys=['attributes', 'retrieved_attributes']),
    dict(type='Collect', keys=['img', 'attributes', 'retrieved_images', 'retrieved_attributes', 'retrieved_labels'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=12,
    train=dict(
        type='RAT',
        data_prefix=data_prefix,
        ann_file='',
        pipeline=train_pipeline),
    val=dict(
        type='RAT',
        data_prefix=data_prefix,
        ann_file='',
        pipeline=test_pipeline),
    test=dict(
        type='RAT',
        data_prefix=data_prefix,
        ann_file='',
        pipeline=test_pipeline))