# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=600, max_keep_ckpts=1)
evaluation = dict(by_epoch=False,
                  metric=None,
                  interval=600)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
