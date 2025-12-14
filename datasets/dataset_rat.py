import os.path as osp
import numpy as np
import random
import copy
from mmcls.datasets import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


@DATASETS.register_module()
class RAT(BaseDataset):
    CLASSES = [str(i) for i in range(3)]

    def __init__(self,
                 patient_level=False,
                 slide_balance=False,
                 patch_balance=False,
                 feature_file=None,
                 num_aug=None,
                 text_embedding_file='your path of text embedding of 3 classes',
                 **kwargs):
        # Remove unused patient_level, slide_balance, patch_balance logic
        # Keep only the basic loading functionality
        self.feature_file = feature_file
        self.num_aug = num_aug
        self.text_embedding = np.load(text_embedding_file)  # Load precomputed embeddings
        super().__init__(**kwargs)

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        if self.feature_file is not None:
            assert self.num_aug >= 1
            features = []
            for a in range(self.num_aug):
                feature = np.load(self.feature_file + f'_{a + 1}.npy')
                features.append(feature)
            features = np.array(features)

        data_infos = []
        with open(self.ann_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                items = line.split('\t')
                
                # Each image group has 10 columns: filename, label, attr1...attr8
                # Total column count should be 10*n, where n = 1 (main image) + k (retrieved images)
                if len(items) < 10 or len(items) % 10 != 0:
                    print(f"Line format is incorrect, skip: {line}")
                    continue

                num_segments = len(items) // 10
                # The first segment is the main image
                main_filename = items[0]
                main_label = int(items[1])
                main_attributes = list(map(float, items[2:10]))

                # Remaining segments contain retrieved image info
                retrieved_filenames = []
                retrieved_labels = []
                retrieved_attributes = []

                for seg_idx in range(1, num_segments):
                    start = seg_idx * 10
                    r_filename = items[start]
                    r_label = int(items[start + 1])
                    r_attrs = list(map(float, items[start+2:start+10]))

                    retrieved_filenames.append(osp.join(self.data_prefix, r_filename))
                    retrieved_labels.append(r_label)
                    retrieved_attributes.append(r_attrs)

                # Use retrieved_labels to gather embeddings from text_embedding (B, k, 1024)
                # Assume text_embedding is an array or dict indexed by label
                retrieved_labels = np.array(retrieved_labels, dtype=np.int64)
                retrieved_embeddings = self.text_embedding[retrieved_labels]  # shape: (k, embed_dim)

                info = {
                    'img_prefix': None,
                    'img_info': {
                        'filename': osp.join(self.data_prefix, main_filename)
                    },
                    'gt_label': np.array(main_label, dtype=np.int64),
                    'img_text': '',
                    'attributes': main_attributes,  # Attributes of the main image
                    'retrieved_images': retrieved_filenames,   # Retrieved image path list
                    'retrieved_labels': retrieved_embeddings,   # Text embeddings of retrieved images
                    'retrieved_attributes': np.array(retrieved_attributes, dtype=np.float32)  # Retrieved image attributes
                }

                if self.feature_file is not None:
                    info['feature'] = features[:, i]

                data_infos.append(info)

        return data_infos


    def __getitem__(self, idx):
        # Balancing logic not needed; return processed result directly
        return self.prepare_data(idx)

    def evaluate(self,
                results,
                metric=None,  # None means compute all supported metrics
                metric_options=None,
                indices=None,
                logger=None,
                save_predictions=True,  # Control whether to save predictions and labels
                save_path=''):  # Save path

        if metric_options is None:
            metric_options = dict()

        # Default metrics
        allowed_metrics = ['accuracy', 'recall', 'f1']
        if metric is None:
            metrics = allowed_metrics
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric

        # Check for unsupported metrics
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if invalid_metrics:
            raise ValueError(f'Metric {invalid_metrics} not supported. Choose from {allowed_metrics}.')

        # Stack results into (N, num_classes)
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]

        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, \
            'Length of results is inconsistent with ground-truth labels.'

        # Predicted class is the index of the max value in each row
        pred_labels = np.argmax(results, axis=1)

        # Compute overall accuracy
        overall_acc = accuracy(results, gt_labels, topk=1).item()

        # Initialize evaluation result dictionary
        eval_results = {}
        if 'accuracy' in metrics:
            eval_results['accuracy'] = overall_acc

        # Compute confusion matrix
        num_classes = len(self.CLASSES)
        confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pred in zip(gt_labels, pred_labels):
            confusion_mat[gt, pred] += 1

        # Compute recall and F1 for each class
        if 'recall' in metrics or 'f1' in metrics:
            for i in range(num_classes):
                TP = confusion_mat[i, i]
                FN = np.sum(confusion_mat[i, :]) - TP
                FP = np.sum(confusion_mat[:, i]) - TP

                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)

                if 'recall' in metrics:
                    eval_results[f'recall_{i}'] = recall
                if 'f1' in metrics:
                    eval_results[f'f1_{i}'] = f1

        # Optionally save predicted and true labels
        if save_predictions:
            self._save_predictions(pred_labels, gt_labels, save_path)

        return eval_results

    def _save_predictions(self, pred_labels, gt_labels, save_path):
        """
        Save predicted and true labels to a CSV file.

        Args:
            pred_labels (np.ndarray): Predicted labels.
            gt_labels (np.ndarray): Ground-truth labels.
            save_path (str): File path to save.
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Predicted_Label': pred_labels,
            'True_Label': gt_labels
        })

        # Optionally add more information, such as filenames
        # For example:
        # filenames = [info['img_info']['filename'] for info in self.data_infos]
        # df['Filename'] = filenames
        # Ensure filenames align with pred_labels and gt_labels

        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f'Predicted and true labels saved to {save_path}')
