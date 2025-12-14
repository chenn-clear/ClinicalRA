from typing import Literal, List
import torch
import torch.nn as nn
import clip
from mmcls.models.builder import BACKBONES, HEADS, NECKS
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.heads import ClsHead
from torchvision.models import resnet18
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal, List
import torch
import torch.nn as nn
import clip
from mmcls.models.builder import BACKBONES, HEADS, CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models import ImageClassifier
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.heads import ClsHead, MultiLabelClsHead
from transformers import AutoModel, AutoTokenizer
from typing import List, Literal
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ModifiedResNet18(nn.Module):
    def __init__(self, input_channels=32):
        super(ModifiedResNet18, self).__init__()
        self.inplanes = 64
        
        # 1. Modify input layer: support 32-channel input (32 axial slices)
        # Keep Kaiming's original ResNet 7x7 conv, stride=2, padding=3
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Stack residual layers (ResNet-18 standard: [2, 2, 2, 2])
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        
        # layer4 outputs 512 channels
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 3. Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Remove classification layer (self.fc)
        # Linear(512, 1000) is unnecessary because we only need feature vectors

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem (input processing)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers (feature extraction)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # shape now: [Batch, 512, H/32, W/32]

        # Head (get embedding vector)
        x = self.avgpool(x) # shape now: [Batch, 512, 1, 1]
        x = torch.flatten(x, 1) # shape now: [Batch, 512]
        
        # Return the 512-dim vector directly, skipping FC
        return x


@CLASSIFIERS.register_module()
class ClinicalDataEnrichedImageClassifier(ImageClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def extract_feat(self, img, attributes, stage='neck'):

        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            retrieved_images (list[Tensor] | None): The retrieved similar images, a list of (num_samples, num_channels, *img_shape), the length of the list == k.
            retrieved_labels (list[np]): The labels of retrieved images, the length of the list == k.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.
        """
        x = self.backbone(img, attributes)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)

        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)

        return x
    
    def forward_train(self, img, gt_label, attributes=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                should be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
            retrieved_images (list[Tensor] | None): The retrieved similar images, a list of (num_samples, num_channels, *img_shape), the length of the list == k.
            retrieved_labels (list[np]): The labels of retrieved images, the length of the list == k.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img, attributes)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, img, attributes=None, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img, attributes)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, **kwargs)

        return res


@CLASSIFIERS.register_module()
class RetrievalAugmentedClinicalDataEnrichedImageClassifier(ImageClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def extract_feat(self, img, attributes, retrieved_images, retrieved_attributes, retrieved_labels, stage='neck'):

        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            retrieved_images (list[Tensor] | None): The retrieved similar images, a list of (num_samples, num_channels, *img_shape), the length of the list == k.
            retrieved_labels (list[np]): The labels of retrieved images, the length of the list == k.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.
        """

        x = self.backbone(img, attributes)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x, retrieved_images, retrieved_attributes, retrieved_labels)

        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)

        return x


    def forward_train(self, img, gt_label, attributes=None, retrieved_images=None, retrieved_attributes=None, retrieved_labels=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                should be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
            retrieved_images (list[Tensor] | None): The retrieved similar images, a list of (num_samples, num_channels, *img_shape), the length of the list == k.
            retrieved_labels (list[np]): The labels of retrieved images, the length of the list == k.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img, attributes, retrieved_images, retrieved_attributes, retrieved_labels)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses


    def simple_test(self, img, attributes=None, retrieved_images=None, retrieved_attributes=None, retrieved_labels=None, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img, attributes, retrieved_images, retrieved_attributes, retrieved_labels)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, **kwargs)

        return res


@BACKBONES.register_module() 
class ClinicalDataEnrichedResnetBackbone(BaseBackbone):

    """
    Backbone that uses the standard Kaiming ResNet-18 (modified for 32-channel input)
    and fuses it with attribute features.
    """

    def __init__(self,
                 output_dim=1024,           
                 attr_dim=8,
                 input_channels=32,        # Added: matches the 32 axial slices
                 fix: bool = False,        # When True, load from checkpoint and freeze params
                 proj: bool = False,
                 init_cfg=None,
                 checkpoint_path=None,
                 # Following params kept for interface compatibility; may not be used by ModifiedResNet18
                 layers=[2, 2, 2, 2],
                 heads=None,
                 input_resolution=None,
                 width=None):    
        super().__init__(init_cfg=init_cfg)

        # -----------------------------------------------------------
        # 1. Initialize ModifiedResNet18 as the visual backbone
        # Use the standard ResNet18 you just defined (input channels=32, output=512)
        # -----------------------------------------------------------
        self.model = ModifiedResNet18(input_channels=input_channels)

        # Randomly initialize model parameters when fix is False
        # Note: ModifiedResNet18 already applies Kaiming init internally; this handles extra layers
        if not fix:
            self.model.apply(self._init_weights)

        # Add a Dropout layer after feature extraction
        self.dropout = nn.Dropout(p=0.2)

        # Optional projection layer
        self.proj = nn.Identity()
        if proj:
            self.proj = nn.Linear(512, output_dim)

        # -----------------------------------------------------------
        # 2. Lightweight MLP Fusion for visual features and attributes
        # -----------------------------------------------------------
        # ModifiedResNet18 output is fixed at 512, so input_dim = 512 + attr_dim
        input_dim = 512 + attr_dim 
        dropout_rate = 0.2
        hidden_dims = [256]  # Single hidden layer
        
        layers_list = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers_list.append(nn.Linear(in_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim
        
        # Final mapping back to output_dim (512)
        layers_list.append(nn.Linear(in_dim, output_dim))
        self.mlp_fusion = nn.Sequential(*layers_list)

        # -----------------------------------------------------------
        # 3. Checkpoint Loading & Freezing
        # -----------------------------------------------------------
        # If fix = True and checkpoint_path is provided, load weights from the checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Note: keys may not match if the checkpoint uses the old architecture; map them as needed
            self.load_state_dict(state_dict, strict=False)

        # Set whether parameters are trainable based on fix
        for param in self.parameters():
            param.requires_grad = not fix

    def forward(self, x, attributes):
        """
        Forward pass with visual features and attributes.
        Args:
            x (Tensor): Input images. Shape: (batch_size, 32, H, W)
            attributes (Tensor): Attribute values with shape (batch_size, attr_dim).
        Returns:
            Tensor: Fused features with shape (batch_size, output_dim).
        """
        x = x.float()
        
        # 1. Visual Feature Extraction
        visual_features = self.model(x)  # Output: (batch_size, 512)
        visual_features = self.dropout(visual_features)

        # 2. Attribute Fusion
        attributes = attributes.float()
        # Concatenate: [Batch, 512] + [Batch, attr_dim] -> [Batch, 512 + attr_dim]
        fused_input = torch.cat([visual_features, attributes], dim=1) 
        
        fused_features = self.mlp_fusion(fused_input)  # Output: (batch_size, 512)

        return self.proj(fused_features)

    @staticmethod
    def _init_weights(module):
        """Randomly initialize weights for all layers."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


@NECKS.register_module()
class RetrievalAugmentedProjectionNeck(nn.Module):
    def __init__(self,
                 output_dim=1024,           
                 input_channels=32,        # Added: matches 32 axial slices
                 attr_dim=8,
                 num_heads=8,
                 dropout=0.3,
                 fix=True,
                 proj=False,
                 checkpoint_path=None,
                 # Keep legacy parameter interface to avoid errors, even if unused downstream
                 arch='resnet18',
                 layers=[2, 2, 2, 2],
                 heads=None,
                 input_resolution=None,
                 width=None):
        super().__init__()

        # 1. Instantiate the visual encoder
        # Use the updated AttributeAugmentedCLIPImageBackbone (contains ModifiedResNet18)
        self.visual_encoder = ClinicalDataEnrichedResnetBackbone(
            output_dim=output_dim,
            attr_dim=attr_dim,
            input_channels=input_channels, # Key: pass the 32-channel setting
            fix=fix,
            proj=proj
        )

        # 2. Load pretrained weights
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strict=False allows partial loading (model structure may have been tweaked)
            self.visual_encoder.load_state_dict(state_dict, strict=False)

        # 3. Define a single multi-head attention module
        # embed_dim must equal output_dim (512)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, 
                                          num_heads=num_heads, 
                                          dropout=dropout, 
                                          batch_first=True)

    def forward(self, x, retrieved_images, retrieved_attributes, retrieved_labels):
        """
        Args:
            x: (B, 512) embedding of the primary image (from backbone)
            retrieved_images: (B, k, 32, H, W) retrieved images (32 channels)
            retrieved_attributes: (B, k, attr_dim) attributes of retrieved images
            retrieved_labels: (B, k, 512) retrieved label embeddings (must also be 512-dim)

        Returns:
            x_out: (B, 512) enhanced image features
        """
        B, k, C, H, W = retrieved_images.shape
        # C should be 32
        
        # 1. Flatten batch and k dimensions for backbone processing
        retrieved_images_flat = retrieved_images.view(B * k, C, H, W)  # (B*k, 32, H, W)
        retrieved_attributes_flat = retrieved_attributes.view(B * k, -1)  # (B*k, attr_dim)

        # 2. Extract features of retrieved images
        # Use AttributeAugmentedCLIPImageBackbone 
        # Output shape: (B*k, 512)
        retrieved_img_embeds = self.visual_encoder(retrieved_images_flat, retrieved_attributes_flat)
        
        # Reshape back to: (B, k, 512)
        retrieved_img_embeds = retrieved_img_embeds.view(B, k, -1)

        # 3. Prepare Query for attention
        # x is the current image feature (B, 512) -> Q: (B, 1, 512)
        x_q = x.unsqueeze(1) 

        # 4. Cross-attention
        # Attention requires Q, K, V to share embed_dim; all are 512 here
        
        # First attention pass:
        # Query = current image (x)
        # Key   = retrieved images (retrieved_img_embeds)
        # Value = retrieved labels (retrieved_labels)
        # Meaning: use the current image to find similar retrieved images and aggregate their label info
        o1, _ = self.attn(x_q, retrieved_img_embeds, retrieved_labels)  # (B, 1, 512)

        # Second attention pass:
        # Query = current image (x)
        # Key   = retrieved labels (retrieved_labels)
        # Value = retrieved images (retrieved_img_embeds)
        # Meaning: use the current image to match retrieved labels and aggregate their image info
        o2, _ = self.attn(x_q, retrieved_labels, retrieved_img_embeds)  # (B, 1, 512)

        # 5. Residual connection and fusion
        # x + o1 + o2
        x_out = x_q + o1 + o2  # (B, 1, 512)
        x_out = x_out.squeeze(1)  # (B, 512)

        return x_out


@HEADS.register_module()
class TextEmbeddingHead(ClsHead):

    """Text embedding head."""

    def __init__(self,
                 texts: List[str],
                 text_encoder: dict,
                 temperature: float = 1.0,
                 learnable_t: bool = False,
                 float16: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        dtype = torch.float16 if float16 else torch.float32
        text_encoder = MODELS.build(text_encoder)
        self.weights = text_encoder(texts).type(dtype)
        self.temperature = torch.tensor(temperature, dtype=dtype).to(DEVICE)
        if learnable_t:
            self.temperature = nn.Parameter(self.temperature)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = x[-1]  # cls token
        return x

    def forward(self, x):
        dtype = x.dtype
        x = self.pre_logits(x)
        x = x / x.norm(dim=-1, keepdim=True)
        weights = self.weights / self.weights.norm(dim=-1, keepdim=True)
        t = self.temperature.exp().type(dtype)
        cls_score = t * x @ weights.type(dtype).t()
        return cls_score

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.forward(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, **kwargs):
        cls_score = self.forward(x)
        return super().simple_test(cls_score, **kwargs)
    

@MODELS.register_module()
class CLIPTextEncoder(nn.Module):
    def __init__(self,
                 model: str = 'openai/clip-vit-base-patch16',
                 key: Literal['last_hidden_state', 'get_text_features'] = 'get_text_features',
                 text_feature_file: str = None):
        super().__init__()
        self.key = key
        if text_feature_file is not None:
            self.text_embeddings = torch.load(text_feature_file, map_location='cpu')
            return

        from transformers import CLIPModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).to(DEVICE)

    def forward(self, texts):
        if hasattr(self, 'text_embeddings'):
            return self.text_embeddings.to(DEVICE)

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            if self.key == 'get_text_features':
                text_embeds = self.model.get_text_features(**inputs)  # Most straightforward approach
            elif self.key == 'last_hidden_state':
                text_embeds = self.model.text_model(**inputs).last_hidden_state[:, 0]
            else:
                raise NotImplementedError(f"Unsupported key: {self.key}")

        return text_embeds.to(DEVICE)


@MODELS.register_module()
class BERT(nn.Module):
    """A wrapper of BERT model for text embedding."""

    def __init__(self,
                 model: str = 'michiyasunaga/BioLinkBERT-large',
                 key: Literal['pooler_output', 'last_hidden_state'] = 'pooler_output',
                 text_feature_file: str = None):
        super().__init__()
        self.key = key
        if text_feature_file is not None:
            self.text_embeddings = torch.load(text_feature_file, map_location='cpu')
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def forward(self, texts):
        if hasattr(self, 'text_embeddings'):
            return self.text_embeddings.to(DEVICE)
        texts = [self.tokenizer(t, return_tensors='pt') for t in texts]

        with torch.no_grad():
            if self.key == 'pooler_output':
                text_embeddings = torch.cat([self.model(**inputs)[self.key] for inputs in texts], dim=0)
            elif self.key == 'last_hidden_state':
                # use [CLS] token
                text_embeddings = torch.cat([self.model(**inputs)[self.key][:, 0] for inputs in texts], dim=0)
            else:
                raise NotImplementedError

        return text_embeddings.to(DEVICE)