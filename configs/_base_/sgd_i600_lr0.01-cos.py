runner = dict(type='IterBasedRunner', max_iters=600)
optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.00005  
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    by_epoch=False,
    warmup='constant',
    warmup_by_epoch=False,
    warmup_iters=12,
    warmup_ratio=0.005)
