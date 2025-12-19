

-----

# Clinical Data-Driven Retrieval-Augmented Model for Lung Nodule Malignancy Prediction

This repository contains the implementation code for the MICCAI 2025 paper: **Clinical Data-Driven Retrieval-Augmented Model for Lung Nodule Malignancy Prediction**.

## 1\. Data Preparation

### Dataset Download

Please download the raw CT image data from the official **LIDC-IDRI** collection:

  * **Source:** [The Cancer Imaging Archive (TCIA) - LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)

### Preprocessing

For CT data preprocessing and the extraction of nodule-related annotations, this project relies on the `pylidc` library.

  * **Documentation:** [pylidc Documentation](https://pylidc.github.io/)

Please refer to the official `pylidc` documentation to parse DICOM data, perform nodule clustering, and extract features.

## 2\. Split Files Format

Before running the code, please ensure you have prepared the train/test split files. The files should be in `.txt` format, where each line represents a sample.

**Format Specifications:**

  * **Delimiter:** Tab (`\t`)
  * **Columns:** 10 columns total (1 Path + 1 Label + 8 Attributes)
  * **Attribute Order:** Subtlety, Internal Structure, Calcification, Sphericity, Margin, Lobulation, Spiculation, Texture.

**Single Line Structure:**

```text
<Image_Path>	<Label>	<Attr1>	<Attr2>	<Attr3>	<Attr4>	<Attr5>	<Attr6>	<Attr7>	<Attr8>
```

**Example Content:**

```text
/data/lidc/crops/0001.npy	1	4.0	4.0	1.0	4.0	5.0	1.0	1.0	5.0
/data/lidc/crops/0002.npy	0	2.0	1.0	1.0	3.0	4.0	1.0	1.0	3.0
...
```

**Column Details:**

1.  **Image Path**: Path to the preprocessed image file (e.g., `.npy` or image format).
2.  **Label**: Classification label (e.g., 0 for Benign, 1 for Unsure, 2 for Malignant).
3.  **Attributes**: The following 8 columns are clinical attribute scores:
      * Subtlety
      * Internal structure
      * Calcification
      * Sphericity
      * Margin
      * Lobulation
      * Spiculation
      * Texture

## 3\. Main Requirements

The project requires the following dependencies:

```text
torch==1.13.0
mmcls==0.25.0
transformers
clip
faiss
```

You can install the necessary packages using pip:

```bash
pip install torch==1.13.0
pip install mmcls==0.25.0
pip install transformers clip faiss-gpu  # Use faiss-cpu if you do not have a GPU
pip install openmim
```

## 4\. Training

To start training, use the `mim` tool. Ensure your current directory is in the `PYTHONPATH`.

```bash
PYTHONPATH=.:$PYTHONPATH mim train mmcls <path_to_config_file.py>
```

  * Replace `<path_to_config_file.py>` with the actual path to your configuration file.

## 5\. Acknowledgements

This repository is built upon **MMPretrain** (formerly MMClassification). We thank the OpenMMLab team for their open-source contribution.

  * **MMPretrain:** [https://github.com/open-mmlab/mmpretrain](https://github.com/open-mmlab/mmpretrain)

-----

**We will continuously improve and update this repository.**
