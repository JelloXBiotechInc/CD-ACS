import numpy as np
import math
import cv2
import gc
import tensorflow as tf
from tqdm import tqdm

import os
import sys

from skimage.util import view_as_blocks

def extract_patches_exact_size(image, patch_shape):
    patches = view_as_blocks(image, patch_shape)
    patches = np.moveaxis(patches, 2, 0)[0]
    return patches

def extract_rectangle_patches(image, patch_shape=(1000, 1000)):
    patch_size_h, patch_size_w = patch_shape
    h, w, c = image.shape
    pph, ppw = (h//patch_size_h)*patch_size_h, (w//patch_size_w)*patch_size_w
    
    patches = extract_patches_exact_size(image[:pph, :ppw, :], (patch_size_h, patch_size_w, c))
    
    return patches

def over_extract_rectangle_patches(image, patch_shape=(1000, 1000)):
    patch_size_h, patch_size_w = patch_shape
    h, w, c = image.shape
    pph, ppw = (h//patch_size_h)*patch_size_h, (w//patch_size_w)*patch_size_w
    if pph < h:
        pph += patch_size_h
    if ppw < w:
        ppw += patch_size_w
    
    image = np.pad(image, ((0, pph - h), (0, ppw - w), (0, 0)), 'constant', constant_values=(0))
    
    patches = extract_patches_exact_size(image, (patch_size_h, patch_size_w, c))
    
    return patches

def extract_patches(image, patch_size=1000):
    return extract_rectangle_patches(image, (patch_size, patch_size))

def combine_patches(patches):
    rows, cols, h, w, c = patches.shape
    expanded = np.expand_dims(patches, axis=2)
    trans = expanded.transpose((0,3,1,4,2,5))
    reshaped = trans.reshape((rows*h, cols*w, c))
    image = reshaped
    return image

def extract_patches_numpy_func(image, crop_size):
    return np.concatenate(extract_rectangle_patches(image, crop_size), axis=0)

@tf.function(input_signature=[tf.TensorSpec((None, None, 3), tf.float32), tf.TensorSpec((2,), tf.int64)])
def extract_channel_3(image, crop_size):
    y = tf.numpy_function(extract_patches_numpy_func, [image, crop_size], tf.float32)
    return y

@tf.function(input_signature=[tf.TensorSpec((None, None, 1), tf.float32), tf.TensorSpec((2,), tf.int64)])
def extract_channel_1(image, crop_size):
    y = tf.numpy_function(extract_patches_numpy_func, [image, crop_size], tf.float32)
    return y

def over_extract_patches_numpy_func(image, crop_size):
    return np.concatenate(over_extract_rectangle_patches(image, crop_size), axis=0)

@tf.function(input_signature=[tf.TensorSpec((None, None, 3), tf.float32), tf.TensorSpec((2,), tf.int64)])
def over_extract_channel_3(image, crop_size):
    y = tf.numpy_function(over_extract_patches_numpy_func, [image, crop_size], tf.float32)
    return y

@tf.function(input_signature=[tf.TensorSpec((None, None, 1), tf.float32), tf.TensorSpec((2,), tf.int64)])
def over_extract_channel_1(image, crop_size):
    y = tf.numpy_function(over_extract_patches_numpy_func, [image, crop_size], tf.float32)
    return y

def get_patches_single(input_image, crop_size=(1000, 1000)):
    channel = list(input_image.shape)[-1]
    shape = [None] + list(crop_size) + [channel]
    if channel == 1:
        input_images = extract_channel_1(input_image, crop_size)
    elif channel == 3:
        input_images = extract_channel_3(input_image, crop_size)
    else:
        raise ValueError(f'Channel {channel} not supported')
    input_images.set_shape(shape)
    
    return input_images

def get_over_patches_single(input_image, crop_size=(1000, 1000)):
    channel = list(input_image.shape)[-1]
    shape = [None] + list(crop_size) + [channel]
    if channel == 1:
        input_images = over_extract_channel_1(input_image, crop_size)
    elif channel == 3:
        input_images = over_extract_channel_3(input_image, crop_size)
    else:
        raise ValueError(f'Channel {channel} not supported')
    input_images.set_shape(shape)
    
    return input_images

def tensors_processed_by_dataset_operations(tensors, dataset_callback=None):
    if dataset_callback == None:
        return tensors
    else:
        length = tensors.shape[0]

        ds = tf.data.Dataset.from_tensor_slices(tensors)
        ds = dataset_callback(ds)

        patched_tensors = []
        for i, patch in enumerate(ds):
            patched_tensors.append(patch)
        return tf.stack(patched_tensors, axis=0)

def predict_image_with_exactly_patched_dataset_operations(
    model,
    image_tensor,
    patch_size,
    batch_size=10,
    before_patch_dataset_callback=None,
    before_dataset_callback=None,
    after_prediction_callback=None,
    after_dataset_callback=None,
    channel=1,
):
    image_tensors = tensors_processed_by_dataset_operations(image_tensor[tf.newaxis, ...], before_patch_dataset_callback)

    image_size = image_tensors.shape
    image_size = np.array(image_size) // np.array([1] + list(patch_size) + [channel])

    def large_patches_single(img):
        return get_patches_single(img, patch_size)

    patches = tensors_processed_by_dataset_operations(image_tensors, lambda ds: ds \
        .map(large_patches_single, num_parallel_calls=1) \
        .unbatch()
    )
    
    batched_patches = before_dataset_callback(tf.data.Dataset.from_tensor_slices(patches)).batch(batch_size)
    predicts = None
    for _patches in batched_patches:
        predict = model.predict(_patches, verbose=0)
    
        if after_prediction_callback != None:
            predict = after_prediction_callback(predict)
        
        if type(predicts) == type(None):
            predicts = predict
        else:
            predicts = tf.stack(tf.unstack(predicts, axis=0) + tf.unstack(predict, axis=0), axis=0)
    
    batched_patches = None
    gc.collect()
    
    patched_imgs = tensors_processed_by_dataset_operations(patches, after_dataset_callback)
    patches = None
    patched_predicts = tensors_processed_by_dataset_operations(predicts, after_dataset_callback)
    predicts = None
    gc.collect()

    reshape_size = np.array(list(image_size[1:3]) + list(patch_size) + [channel])

    patched_imgs = tf.stack(patched_imgs, axis=0)
    patched_imgs = tf.reshape(patched_imgs, reshape_size)
    patches = combine_patches(patched_imgs)
    patched_imgs = None
    gc.collect()

    reshape_size = np.array(list(image_size[1:3]) + list(patch_size) + [1])
    
    patched_predicts = tf.stack(patched_predicts, axis=0)
    patched_predicts = tf.reshape(patched_predicts, reshape_size)
    predicts = combine_patches(patched_predicts)
    patched_predicts = None
    gc.collect()

    return patches, predicts

def predict_image_with_patched_dataset_operations(
    model,
    image_tensor,
    patch_size,
    batch_size=10,
    *args,
    **kwargs,
):
    height, width, channel = image_tensor.shape
    shape = np.array([height, width])
    crop = np.array(patch_size)
    patched_shape = shape // crop * crop

    callback = lambda img: predict_image_with_exactly_patched_dataset_operations(
        model,
        img,
        patch_size,
        batch_size=batch_size,
        *args,
        **kwargs,
    )

    if (shape == patched_shape).all():
        img, predict = callback(image_tensor)
    else:
        enlarged_shape = patched_shape + crop
        enlarged_image_tensor = tf.pad(image_tensor, tf.constant([
            [0, enlarged_shape[0] - shape[0]],
            [0, enlarged_shape[1] - shape[1]],
            [0, 0],
        ]), "CONSTANT")
        img, predict = callback(enlarged_image_tensor)
        img = img[:shape[0], :shape[1], :]
        predict = predict[:shape[0], :shape[1], :]

    return img, predict

from . import CD_Lossfunction
CD = CD_Lossfunction

class PatchLargeImage:
    BATCH_SIZE = 4
    CD = None
    model = None
    batch_callback = None
    
    def __init__(self, _CD, model, batch_callback=None):
        self.CD = _CD if _CD != None else CD
        self.model = model
        self.batch_callback = batch_callback if batch_callback else self.cd_batch_callback

    def cd_batch_callback(self, batch):
        cd_batch = np.zeros((*list(batch.shape)[:3], 1))

        for i in range(list(batch.shape)[0]):
            img = batch[i, :, :, :]

            hed_from_rgb = np.linalg.inv(self.CD.rgb_from_hed)
            stains = self.CD.separate_stains(img, hed_from_rgb)
            cd_batch[i, :, :, :] = np.reshape(stains[:, :, 0], (*list(stains.shape)[:2], 1))

            hed_from_rgb = None
            stains = None
            gc.collect()
        return cd_batch
    
    def r_channel_batch_callback(self, batch):
        r_batch = np.zeros((*list(batch.shape)[:3], 1))

        for i in range(list(batch.shape)[0]):
            img = batch[i, :, :, :]
            
            r_batch[i, :, :, :] = np.expand_dims(img[:, :, 0], axis=-1)

            img = None
            gc.collect()
        return r_batch
    
    def subdiv_already_subdiv_np_images(self, img_batch_subdiv_np):
        subdiv = self.BATCH_SIZE
        total_per_batch = list(img_batch_subdiv_np.shape)[0]
        factor = total_per_batch // subdiv
        remainder = total_per_batch - factor * subdiv
        split_counts = [subdiv]*factor + ([remainder] if remainder != 0 else [])
        split_starts = [np.sum(split_counts[:i+1]) for i, c in enumerate(split_counts[:-1])]
        img_batch_subdiv_np = np.split(img_batch_subdiv_np, split_starts, axis=0)

        mod_img_batch_subdiv_np = []
        for batch in img_batch_subdiv_np:
            batch = self.batch_callback(batch)
            
            mod_img_batch_subdiv_np.append(batch)

            batch = None
            gc.collect()

        img_batch_subdiv_np = None
        gc.collect()

        preds = []
        for batch in mod_img_batch_subdiv_np:
            batch_data = tf.data.Dataset.from_tensor_slices(np.array([batch]))
            
            pred = self.model.predict(batch_data, verbose=0)

            batch_data = None
            gc.collect()

            preds.append(pred)

            pred = None
            gc.collect()
        
        input_batch_subdiv_np = np.concatenate(mod_img_batch_subdiv_np, axis=0)
        output_batch_subdiv_np = np.concatenate(preds, axis=0)

        preds = None
        gc.collect()

        return input_batch_subdiv_np, output_batch_subdiv_np

    def predict_cpu(
        self,
        input_img,
        patch_size=1000,
        resize_size=1024
    ):
        h, w, c = input_img.shape
        dh = math.ceil(h / patch_size) * patch_size
        dw = math.ceil(w / patch_size) * patch_size
        input_img = np.pad(input_img, ((0, dh-h), (0, dw-w), (0, 0)), mode='constant')
        
        resize = lambda img, resize_size: cv2.resize(img, dsize=(resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
        
        patches = extract_patches(input_img, patch_size)
        input_img = None
        gc.collect()
        
        input_results = []
        results = []
        rows, cols, _, _, _ = patches.shape
        for row in range(rows):
            patch = patches[row]
            imgs = []
            for col in range(cols):
                img = resize(patch[col], resize_size)
                imgs.append(img)
            
            input_row_results, row_results = self.subdiv_already_subdiv_np_images(np.stack(imgs, axis=0))
            
            resized_input_row_results = []
            for input_row_result in input_row_results:
                if c == 1:
                    resized_input_row_results.append(np.expand_dims(resize(input_row_result, patch_size), axis=-1))
                else:
                    resized_input_row_results.append(resize(input_row_result, patch_size))
            
            resized_row_results = []
            for row_result in row_results:
                resized_row_results.append(resize(row_result, patch_size))
            
            input_results.append(np.stack(resized_input_row_results, axis=0))
            results.append(np.stack(resized_row_results, axis=0))
        patches = None
        gc.collect()
        
        patched_input_result = combine_patches(np.stack(input_results, axis=0))[:h, :w, :]
        input_results = None
        gc.collect()
        
        patched_results = combine_patches(np.stack(results, axis=0))
        results = None
        gc.collect()
        
        patched_pred_result = np.expand_dims(np.argmax(patched_results, axis=-1), axis=-1)[:h, :w, :]
        patched_results = None
        gc.collect()
        
        return patched_input_result, patched_pred_result

    def predict_gpu(
        self,
        input_img,
        patch_size=(1000, 1000),
        batch_size=10,
        *args,
        **kwargs,
    ):
        patched_input_result, patched_pred_result = predict_image_with_patched_dataset_operations(
            self.model,
            tf.convert_to_tensor(input_img),
            patch_size,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        
        return patched_input_result, patched_pred_result