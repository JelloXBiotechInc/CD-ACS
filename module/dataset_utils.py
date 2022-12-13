import os, sys

from .model_utils import Dataset
from .patch_large_image import extract_patches, combine_patches

import tensorflow as tf
import numpy as np
from skimage.exposure import match_histograms

class EmptyDatapointWrapper(Dataset.DatapointWrapper):
    def __call__(self, *datapoint):
        datapoint = self.callback(*datapoint)
        
        return datapoint

class BasicDatasetProcess:
    def __init__(self, dataset, dataset_size, num_parallel_calls=1):
        # self.dataset = dataset
        self.dataset_size = dataset_size
        self.processed_dataset = dataset
        
        self.num_parallel_calls = num_parallel_calls
        self.datasetWrapper = EmptyDatapointWrapper

    def __repr__(self):
        return self.processed_dataset.__repr__() + "{ dataset_size:" + str(self.dataset_size) + "}"

    def __str__(self):
        return self.processed_dataset.__str__() + "{ dataset_size:" + str(self.dataset_size) + "}"
    
    def new(self, dataset, dataset_size):
        new_obj = self.__class__(dataset, dataset_size, self.num_parallel_calls)
        return new_obj
    
    def change_num_parallel_calls(self, new_num_parallel_calls):
        self.num_parallel_calls = new_num_parallel_calls
        return self
    
    def normalize(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.normalize
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def convert_seg_type(self, dtype=tf.float32):
        convert_type = lambda img, mask: (img, tf.cast(mask, dtype))
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                convert_type
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def h_channel(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.color_deconv_h_channel
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def he_channel(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.color_deconv_he_channel
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def cd_normalize(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.color_deconv_normalize
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def histogram_matching(self, ref_img):
        def histo_match_numpy_func(image):
            return match_histograms(image, ref_img, channel_axis=-1)

        @tf.function(input_signature=[tf.TensorSpec((None, None, None), tf.float32)])
        def histo_match_tf_fn(image):
            y = tf.numpy_function(histo_match_numpy_func, [image], tf.float32)
            return y
        
        def histo_match(input_image, input_mask):
            
            input_image = histo_match_tf_fn(input_image)
            
            return input_image, input_mask
        
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                histo_match
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def map_ratio(self, ratio):
        def map_ratio_single(image):
            image = image * ratio
            image = tf.clip_by_value(image, 0, 1)
            return image
        
        def _map_ratio(input_image, input_mask):
            input_image = map_ratio_single(input_image)
            
            return input_image, input_mask
        
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                _map_ratio
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def r_channel(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.r_channel
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def rg_channel(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.rg_channel
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def pad_input_channel_to_3(self):
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                Dataset.pad_input_channel_to_3
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def resize_image(self, patch_size):
        resize_image_callback = lambda img, mask: Dataset.resize_image(img, mask, patch_size)
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                resize_image_callback
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def to_categorical(self):
        to_categorical_callback = lambda img, mask: Dataset.to_categorical(img, mask)
        processed_dataset = self.processed_dataset.map(
            self.datasetWrapper(
                to_categorical_callback
            ),
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)
    
    def random_patches(self, num_patches, patch_size):
        dataset_size = self.dataset_size * num_patches
        
        random_patches = lambda img, mask: Dataset.random_patches(img, mask, num_patches, patch_size)
        processed_dataset = self.processed_dataset.map(
            random_patches,
            num_parallel_calls=self.num_parallel_calls
        ).unbatch()
        return self.new(processed_dataset, dataset_size)
    
    def extract_large_patches(self, patch_size):
        crop_size = (patch_size, patch_size)
        
        def extract_patches_numpy_func(image):
            return np.concatenate(extract_patches(image, patch_size), axis=0)

        @tf.function(input_signature=[tf.TensorSpec((None, None, None), tf.float32)])
        def extract_large(image):
            y = tf.numpy_function(extract_patches_numpy_func, [image], tf.float32)
            return y
        
        def _extract_large_patches(input_image, input_mask):
            c1 = list(input_image.shape)[-1]
            c2 = list(input_mask.shape)[-1]
            
            shape = [None] + list(crop_size)

            input_images = extract_large(input_image)
            input_images.set_shape(shape + [c1])
            input_masks = extract_large(input_mask)
            input_masks.set_shape(shape + [c2])

            return input_images, input_masks
        
        TOTAL_LENGTH = 0
        for image, mask in self.processed_dataset.take(self.dataset_size):
            image_size = image.shape
            image_size = np.ceil(np.array(image_size) / patch_size).astype(np.int32)
            TOTAL_LENGTH += image_size[0] * image_size[1]
        dataset_size = TOTAL_LENGTH
        
        processed_dataset = self.processed_dataset.map(
            _extract_large_patches,
            num_parallel_calls=self.num_parallel_calls
        ).unbatch()
        return self.new(processed_dataset, dataset_size)
    
    def concatenate(self, other_dataset_process):
        processed_dataset = self.processed_dataset.concatenate(other_dataset_process.processed_dataset)
        dataset_size = self.dataset_size + other_dataset_process.dataset_size
        
        return self.new(processed_dataset, dataset_size)
    
    def random_split(self, percentage, shuffle_size=100):
        if percentage > 1:
            raise ValueError
        
        desired_dataset_size = int(self.dataset_size * percentage)

        random_processed_dataset = self.processed_dataset.shuffle(shuffle_size)

        a_dataset = random_processed_dataset.take(desired_dataset_size)
        b_dataset = random_processed_dataset.skip(desired_dataset_size)
        
        return self.new(a_dataset, desired_dataset_size), self.new(b_dataset, self.dataset_size - desired_dataset_size)
    
    def assert_callback(self, callback):
        processed_dataset = callback(self.processed_dataset)
        return self.new(processed_dataset, self.dataset_size)
    
    def iterobjs(self):
        return self.processed_dataset.take(self.dataset_size)
    
    @staticmethod
    def get_dataset_wrapper_from_dataset(dataset_name, *args, **kwargs):
        dataset_obj = Dataset(dataset_name)
        return BasicDatasetProcess.get_dataset_wrapper_from_dataset_obj(dataset_obj, *args, **kwargs)
    
    @staticmethod
    def get_dataset_wrapper_from_dataset_obj(dataset_obj, wrapper_type='HE'):
        if wrapper_type == 'RAW':
            wrapper_type = DatapointDatasetProcess
        elif wrapper_type == 'HE':
            wrapper_type = HE_H_Channel_DatasetProcess
        elif wrapper_type == 'IF':
            wrapper_type = IF_R_Channel_DatasetProcess
        elif wrapper_type == 'HE_HE':
            wrapper_type = HE_HE_Channel_DatasetProcess
        elif wrapper_type == 'IF_RG':
            wrapper_type = IF_RG_Channel_DatasetProcess
        elif wrapper_type == 'HE_CR':
            wrapper_type = HE_H_Channel_CR_DatasetProcess
        elif wrapper_type == 'IF_CR':
            wrapper_type = IF_R_Channel_CR_DatasetProcess
        
        datasets = dataset_obj.dataset
        infos = dataset_obj.info
        keys = datasets.keys()
        dataset_wrappers = {}
        for key in keys:
            dataset_wrappers[key] = wrapper_type(datasets[key], infos.splits[key].num_examples)
        
        return dataset_wrappers

class DatapointDatasetProcess(BasicDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def unpack_datapoint(self):
        processed_dataset = self.processed_dataset.map(
            Dataset.unpack_datapoint,
            num_parallel_calls=self.num_parallel_calls
        )
        return BasicDatasetProcess(processed_dataset, self.dataset_size, self.num_parallel_calls)
    
    def process(self):
        return self \
            .normalize() \
            .convert_seg_type()

class HE_H_Channel_DatasetProcess(DatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .convert_seg_type() \
            .h_channel()

class IF_R_Channel_DatasetProcess(DatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .convert_seg_type() \
            .r_channel()

class HE_HE_Channel_DatasetProcess(DatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .convert_seg_type() \
            .he_channel()

class IF_RG_Channel_DatasetProcess(DatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .convert_seg_type() \
            .rg_channel()

class CRDatapointDatasetProcess(DatapointDatasetProcess):
    def CR_overlap(self):
        def cr_overlap_func(image, mask, cr_mask):
            cr_mask = tf.cast(tf.round(cr_mask / 255), tf.bool)
            mask = tf.math.logical_and(tf.cast(mask, bool), tf.cast(cr_mask, bool))
            
            mask = tf.cast(mask, tf.bool)
            
            return image, mask, cr_mask
        
        def datapoint_callback(datapoint):
            image = datapoint['image']
            mask = datapoint['segmentation_mask']
            cr_mask = datapoint['color_region_mask']

            image, mask, cr_mask = cr_overlap_func(image, mask, cr_mask)

            datapoint['image'] = image
            datapoint['segmentation_mask'] = mask
            datapoint['color_region_mask'] = cr_mask
            return datapoint
        
        processed_dataset = self.processed_dataset.map(
            datapoint_callback,
            num_parallel_calls=self.num_parallel_calls
        )
        return self.new(processed_dataset, self.dataset_size)

class HE_H_Channel_CR_DatasetProcess(CRDatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .CR_overlap() \
            .convert_seg_type() \
            .h_channel()

class IF_R_Channel_CR_DatasetProcess(CRDatapointDatasetProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.datasetWrapper = Dataset.DatapointWrapper
    
    def process(self):
        return self \
            .normalize() \
            .CR_overlap() \
            .convert_seg_type() \
            .r_channel()