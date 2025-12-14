import clip
import random
from PIL import Image
from torchvision import transforms
from mmcls.datasets.builder import PIPELINES
import numpy as np
import torch
from scipy.ndimage import rotate



@PIPELINES.register_module()
class Pipeline_AT:
    """Pipeline for multi-channel 32x32 CT images."""

    def __init__(self, test_mode=False, img_size=(32, 32), normalize=True):
        """
        Args:
            test_mode (bool): If True, apply test transformations only.
            img_size (tuple): Target image size (height, width) for resizing.
            normalize (bool): If True, apply normalization.
        """
        self.test_mode = test_mode
        self.img_size = img_size
        self.normalize = normalize
        self.mean = [0.5] * 32  # Assumed mean for 32-channel data
        self.std = [0.25] * 32  # Assumed std for 32-channel data

    def __call__(self, results):
        """
        Args:
            results (dict): A dictionary containing image data and metadata.

        Returns:
            dict: Processed data with augmented/normalized image tensor.
        """
        # 1. Load the multi-channel CT image
        filename = results['img_info']['filename']
        with open(filename, 'rb') as f:
            img_data = np.load(f)  # Assuming a .npy file containing shape (32, 32, 32)

        # Ensure the input shape is correct
        assert img_data.shape == (32, 32, 32), f"Expected shape (32, 32, 32), got {img_data.shape}"

        # 2. Apply data augmentation (if not in test mode)
        if not self.test_mode:
            # Random horizontal flip
            if random.random() > 0.5:
                img_data = np.flip(img_data, axis=2).copy()  # Ensure strides are valid

            # Random vertical flip
            if random.random() > 0.5:
                img_data = np.flip(img_data, axis=1).copy()  # Ensure strides are valid

            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                img_data = np.rot90(img_data, k=random.randint(1, 3), axes=(1, 2)).copy()  # Ensure strides are valid

        # 3. Resize to target size (if required)
        if self.img_size != (32, 32):
            img_data_resized = []
            for c in range(img_data.shape[0]):  # Resize each channel individually
                resized_channel = cv2.resize(img_data[c], self.img_size, interpolation=cv2.INTER_LINEAR)
                img_data_resized.append(resized_channel)
            img_data = np.stack(img_data_resized, axis=0)

        # 4. Normalize the image data
        if self.normalize:
            # Convert to float32 if not already
            img_data = img_data.astype(np.float32)

            for c in range(img_data.shape[0]):  # Normalize each channel independently
                img_data[c] = (img_data[c] - self.mean[c]) / self.std[c]

        # Ensure the array is contiguous before converting to tensor
        img_data = np.ascontiguousarray(img_data)

        # Convert to PyTorch tensor
        img_tensor = torch.tensor(img_data, dtype=torch.float32)

        # 6. Update the results dictionary
        results['filename'] = filename
        results['img'] = img_tensor  # Tensor shape: (32, H, W)

        return results


@PIPELINES.register_module()
class Pipeline_RAT:
    """Pipeline for multi-channel 32x32 CT images."""

    def __init__(self, test_mode=False, img_size=(32, 32), normalize=True):
        """
        Args:
            test_mode (bool): If True, apply test transformations only.
            img_size (tuple): Target image size (height, width) for resizing.
            normalize (bool): If True, apply normalization.
        """
        self.test_mode = test_mode
        self.img_size = img_size
        self.normalize = normalize
        self.mean = [0.5] * 32  
        self.std = [0.25] * 32  

    def __call__(self, results):
        """
        Args:
            results (dict): A dictionary containing image data and metadata.

        Returns:
            dict: Processed data with augmented/normalized image tensor.
        """
        # 1. Load the multi-channel CT image
        filename = results['img_info']['filename']
        with open(filename, 'rb') as f:
            img_data = np.load(f)  # shape (32, 32, 32)

        assert img_data.shape == (32, 32, 32), f"Expected shape (32, 32, 32), got {img_data.shape}"

        # 2. Data augmentation (if not test mode)
        if not self.test_mode:
            # Random horizontal flip
            if random.random() > 0.5:
                img_data = np.flip(img_data, axis=2).copy()
            # Random vertical flip
            if random.random() > 0.5:
                img_data = np.flip(img_data, axis=1).copy()
            # Random rotation
            if random.random() > 0.5:
                img_data = np.rot90(img_data, k=random.randint(1, 3), axes=(1, 2)).copy()

        # 3. Resize if needed
        if self.img_size != (32, 32):
            img_data_resized = []
            for c in range(img_data.shape[0]):
                resized_channel = cv2.resize(img_data[c], self.img_size, interpolation=cv2.INTER_LINEAR)
                img_data_resized.append(resized_channel)
            img_data = np.stack(img_data_resized, axis=0)

        # 4. Normalize
        if self.normalize:
            img_data = img_data.astype(np.float32)
            for c in range(img_data.shape[0]):
                img_data[c] = (img_data[c] - self.mean[c]) / self.std[c]

        img_data = np.ascontiguousarray(img_data)
        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        results['filename'] = filename
        results['img'] = img_tensor  # shape: (32, H, W)
        
        # Process retrieved images if they exist
        # retrieved_images: (k, 32, 32, 32)
        if 'retrieved_images' in results and 'retrieved_labels' in results:
            processed_retrieved_imgs = []
            for fname in results['retrieved_images']:
                with open(fname, 'rb') as f:
                    rimg = np.load(f)  # (32,32,32)
                
                # Data preprocessing...
                # Convert to tensor after preprocessing
                rimg = rimg.astype(np.float32)
                for c in range(rimg.shape[0]):
                    rimg[c] = (rimg[c] - self.mean[c]) / self.std[c]

                rimg_tensor = torch.tensor(rimg, dtype=torch.float32)
                processed_retrieved_imgs.append(rimg_tensor)
            # Remove retrieved_images list when it is no longer needed
            del results['retrieved_images']
            # Combine into (k, 32, H, W)
            retrieved_imgs_tensor = torch.stack(processed_retrieved_imgs, dim=0)
            results['retrieved_images'] = retrieved_imgs_tensor


            # Convert retrieved_labels to tensor
            retrieved_embeddings = results['retrieved_labels']
            results['retrieved_labels'] = torch.tensor(retrieved_embeddings, dtype=torch.float32)

        return results
