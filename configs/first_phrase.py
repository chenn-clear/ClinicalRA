_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/iter_based_runtime.py',
    '../_base_/dataset_first_phrase.py',
    '../_base_/sgd_i600_lr0.01-cos.py'
]

lr = 1e-2
run_name = f'first_phrase'

TEXTS = [
    "benign pulmonary nodule",       
    "indeterminate pulmonary nodule",  
    "malignant pulmonary nodule"   
]

model = dict(
    type='RetrievalAugmentedClinicalDataEnrichedImageClassifier',
    backbone=dict(type='ClinicalDataEnrichedResnetBackbone', checkpoint_path = None),
    head=dict(
        type='TextEmbeddingHead',
        texts=TEXTS,
        temperature=4.6052,
        float16=True,
        text_encoder=dict(
            type='BERT',
            model='michiyasunaga/BioLinkBERT-large')))

optimizer = dict(lr=lr)

work_dir = f'work_dirs/{run_name}'