import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm

from . import pix2pix
from .patch_large_image import get_patches_single, get_over_patches_single, PatchLargeImage

class Model:
    model = None
    
    def __init__(self, output_channels:int, input_channels:int=1, input_size:int=1024):
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.input_size = input_size
    
    def easy_predict_single(
        self,
        image,
        patch_size=1000,
        batch_size=10,
        after_callback=None,
        after_prediction_callback=None,
        display_imgs=False,
        save_imgs=False,
        output_dir=None,
        verbose=False,
    ):
        class PridictionCallback:
            pred_mask = None

            def __call__(
                self,
                file_name,
                image,
                mask,
                processed_input_image,
                pred_mask,
                timer,
            ):
                self.pred_mask = pred_mask
                if after_callback != None:
                    after_callback(
                        file_name,
                        image,
                        mask,
                        processed_input_image,
                        pred_mask,
                        timer,
                    )
        
        records = [{
            'file_name': '',
            'image': image,
            'segmentation_mask': [],
        }]
        dataset = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(records).to_dict(orient="list"))
        after_predict_callback = PridictionCallback()

        self.easy_predict(
            dataset,
            patch_size,
            1,
            batch_size=batch_size,
            after_callback=after_predict_callback,
            after_prediction_callback=after_prediction_callback,
            display_imgs=display_imgs,
            save_imgs=save_imgs,
            output_dir=output_dir,
            verbose=verbose,
        )
        return after_predict_callback.pred_mask
    
    def easy_predict(
        self,
        dataset,
        patch_size,
        dataset_length,
        batch_size=10,
        after_callback=None,
        after_prediction_callback=None,
        display_imgs=True,
        save_imgs=False,
        output_dir=None,
        verbose=False,
    ):
        crop_size = (patch_size, patch_size)

        if self.model == None:
            raise ValueError('Doesn\'t assign model yet.')
        
        patch_obj = PatchLargeImage(None, self.model)
        num_parallel_calls = 1
        
        before_patch_dataset_callback = None
        channel_size = self.input_channels
        
        def display_large(display_list):
            plt.figure(figsize=(30, 30))

            title = ['Input Image', 'True Mask', 'Predicted Mask']

            for i in range(len(display_list)):
                gray_args = {
                    'cmap': 'gray',
                } if i == 0 else {}

                plt.subplot(1, len(display_list), i+1)
                plt.title(title[i])
                plt.imshow(display_list[i], **gray_args)
                plt.axis('off')
            plt.show()

        def resize_before_predict(img):
            return Dataset.resize_image_rectangle_single(img, (self.input_size, self.input_size))

        def resize_after_predict(img):
            return Dataset.resize_image_rectangle_single(img, crop_size)

        class Delta:
            start = None
            def delta(self):
                now = time.time()
                if self.start == None:
                    self.start = now
                    return 0
                else:
                    d = now - self.start
                    self.start = now
                    return d
        iterator = range(dataset_length)
        if verbose:
            iterator = tqdm(iterator)
        
        for i in iterator:
            iterator = dataset.skip(i).take(1)
            for obj in iterator:
                file_name = obj['file_name'].numpy().decode('utf-8')

                image, mask = obj['image'], obj['segmentation_mask']

                if verbose:
                    print(f"Start prediction with image : {file_name} ...")
                timer = Delta()
                timer.delta()
                
                if after_prediction_callback == None:
                    after_prediction_callback = lambda predicts: tf \
                        .argmax(predicts, axis=-1, output_type=tf.dtypes.uint16) \
                        [..., tf.newaxis]

                processed_input_image, pred_mask = patch_obj.predict_gpu(
                    image,
                    (patch_size, patch_size),
                    batch_size=batch_size,
                    before_patch_dataset_callback=before_patch_dataset_callback,
                    before_dataset_callback=lambda ds: ds \
                        .map(resize_before_predict, num_parallel_calls=num_parallel_calls)
                    ,
                    after_prediction_callback=after_prediction_callback,
                    after_dataset_callback=lambda ds: ds \
                        .map(resize_after_predict, num_parallel_calls=num_parallel_calls)
                    ,
                    channel=channel_size,
                )
                if verbose:
                    print(f" - End prediction with time : {timer.delta()}")
                
                if display_imgs:
                    if verbose:
                        print("Start displaying imgs...")
                    display_list = [processed_input_image, mask, pred_mask]
                    display_large(display_list)
                    if verbose:
                        print(f" - End displaying imgs with time : {timer.delta()}")
                   
                if save_imgs and output_dir != None:
                    if verbose:
                        print("Start saving imgs...")
                    def saveimg(img, tp):
                        if str(type(img)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
                            img = img.numpy()
                        output_file_name = os.path.join(output_dir, f"{file_name.split('.')[0]}_{tp}.png")

                        height, width, channel = img.shape
                        if channel == 1:
                            plt.imsave(output_file_name, img.reshape((height, width)), cmap='gray')
                        else:
                            plt.imsave(output_file_name, img)

                    saveimg(processed_input_image, 'input')
                    saveimg(mask, 'gt')
                    saveimg(pred_mask, 'pred')
                    if verbose:
                        print(f" - End saving imgs with time : {timer.delta()}")
                
                if after_callback != None:
                    after_callback(
                        file_name,
                        image,
                        mask,
                        processed_input_image,
                        pred_mask,
                        timer,
                    )
    
    def predict_cpu(
        self,
        input_img,
        patch_size,
        h_channel=False,
    ):
        patch_obj = PatchLargeImage(None, self.model, None if h_channel else lambda x: x)
        img, result = patch_obj.predict_cpu(input_img, patch_size=patch_size, resize_size=self.input_size)
        return img, result

class MobileNetV2_1024_Model(Model):
    def __init__(self, output_channels:int, input_channels:int=1, input_size:int=1024):
        super().__init__(output_channels, input_channels, input_size)
        
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

        # Use the activations of these layers
        self.layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        self.base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model_outputs)
        
        # # Refactor to dynamic upsampling based on input_kernel_size
        # self.up_stack = []
        # start_kernel_size = 64
        # while start_kernel_size*2 <= input_size:
        #     self.up_stack.insert(0, pix2pix.upsample(start_kernel_size, 3))
        #     start_kernel_size *= 2
        
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        
        self.model = self.unet_with_mobileNetV2_as_backbone_model(self.output_channels, self.input_channels, self.input_size)

    def unet_with_mobileNetV2_as_backbone_model(self, output_channels:int, input_channels:int, input_size:int):
        inputs = tf.keras.layers.Input(shape=[input_size, input_size, input_channels])
        x = inputs

        x = tf.image.resize(x, (224, 224))
        
        if input_channels == 1:
            x = tf.unstack(x, axis=-1)[0]
            x = tf.stack([x, x, x], axis=-1)

        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
          filters=output_channels, kernel_size=3, strides=2,
          padding='same')  #64x64 -> 128x128

        x = last(x)

        x = tf.image.resize(x, (input_size, input_size))
        outputs = x

        return tf.keras.Model(inputs=inputs, outputs=outputs)

od = tf.constant([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [-0.21, -0.05, 0.5945],
])

class Dataset:
    verbose = False
    
    def __init__(self, name):
        self.name = name
        
        dataset, info = tfds.load(name, with_info=True)
        self.dataset = dataset
        self.info = info
    
    class DatapointWrapper:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, datapoint):
            image = datapoint['image']
            mask = datapoint['segmentation_mask']

            image, mask = self.callback(image, mask)

            datapoint['image'] = image
            datapoint['segmentation_mask'] = mask
            return datapoint
    
    @staticmethod
    def unpack_datapoint(datapoint):
        if Dataset.verbose:
            print("- Sart unpack_datapoint")
        input_image = datapoint['image']
        input_mask = datapoint['segmentation_mask']
        if Dataset.verbose:
            print("-- End unpack_datapoint")

        return input_image, input_mask

    @staticmethod
    def normalize_rgb(input_image):
        if Dataset.verbose:
            print("- Sart normalize_rgb")
        
        input_image = tf.cast(input_image, tf.float32) / 255.0
        
        if Dataset.verbose:
            print("-- End normalize_rgb")

        return input_image

    @staticmethod
    def normalize(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart normalize")
        input_image = Dataset.normalize_rgb(input_image)

        # empty_channel = tf.zeros_like(input_mask, tf.bool)
        # input_mask = tf.stack(tf.unstack(empty_channel, axis=-1) + tf.unstack(tf.cast(input_mask // 215, tf.bool), axis=-1), axis=-1)
        input_mask = tf.cast(tf.round(input_mask / 255), tf.bool)
        if Dataset.verbose:
            print("-- End normalize")

        return input_image, input_mask
    
    @staticmethod
    def combine_stains(stains, conv_matrix):
        if Dataset.verbose:
            print("- Sart combine_stains")
        # log_adjust here is used to compensate the sum within separate_stains().
        log_adjust = -tf.math.log(1E-6)
        log_rgb = -tf.matmul((stains * log_adjust),  conv_matrix)
        rgb = tf.math.exp(log_rgb)
        if Dataset.verbose:
            print("-- End combine_stains")

        return tf.clip_by_value(rgb, 0, 1)

    @staticmethod
    def separate_stains(rgb, conv_matrix):
        if Dataset.verbose:
            print("- Sart separate_stains")
        rgb = tf.math.maximum(rgb, 1E-6)  # avoiding log artifacts
        log_adjust = tf.math.log(1E-6)  # used to compensate the sum above
        stains = tf.matmul((tf.math.log(rgb) / log_adjust), conv_matrix)
        stains = tf.math.maximum(stains, 0)
        if Dataset.verbose:
            print("-- End separate_stains")

        return stains

    @staticmethod
    def rgb2hed(rgb):
        return Dataset.separate_stains(rgb, tf.linalg.inv(od))

    @staticmethod
    def hed2rgb(hed):
        return Dataset.combine_stains(hed, od)

    @staticmethod
    def hed_channel2rgb(hed, axis=0):
        channels = tf.unstack(hed, axis=-1)
        c = channels[axis]
        e = tf.zeros_like(c)
        rgb = Dataset.combine_stains(tf.stack([c, e, e], axis=-1), od)
        return rgb

    @staticmethod
    def hed_channel_single(input_image, axis=0):
        if Dataset.verbose:
            print("- Sart hed_channel_single")
        
        hed = Dataset.rgb2hed(tf.cast(input_image, tf.float32))
        channels = tf.unstack(hed, axis=-1)

        input_image = channels[axis][..., tf.newaxis]
        
        if Dataset.verbose:
            print("-- End hed_channel_single")

        return input_image

    @staticmethod
    def color_deconv_h_channel(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart color_deconv_h_channel")
        
        input_image = Dataset.hed_channel_single(input_image, axis=0)
        
        if Dataset.verbose:
            print("-- End color_deconv_h_channel")

        return input_image, input_mask
    
    @staticmethod
    def color_deconv_normalize_single(input_image):
        if Dataset.verbose:
            print("- Sart color_deconv_normalize_single")
        
        input_image = input_image / tf.math.reduce_sum(tf.linalg.inv(od))
        input_image = tf.clip_by_value(input_image, 0, 1)
        
        if Dataset.verbose:
            print("-- End color_deconv_normalize_single")

        return input_image
    
    @staticmethod
    def color_deconv_normalize(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart color_deconv_normalize")
        
        input_image = Dataset.color_deconv_normalize_single(input_image)
        
        if Dataset.verbose:
            print("-- End color_deconv_normalize")

        return input_image, input_mask

    @staticmethod
    def color_deconv_he_channel(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart color_deconv_he_channel")
        
        h_channel = Dataset.hed_channel_single(input_image, axis=0)[..., 0]
        e_channel = Dataset.hed_channel_single(input_image, axis=1)[..., 0]
        input_image = tf.stack([h_channel, e_channel], axis=-1)
        
        if Dataset.verbose:
            print("-- End color_deconv_he_channel")

        return input_image, input_mask
    
    @staticmethod
    def rgb_channel_single(input_image, axis=0):
        if Dataset.verbose:
            print("- Sart rgb_channel_single")
        
        input_image = tf.stack([input_image[..., axis]], axis=-1)
        
        if Dataset.verbose:
            print("-- End rgb_channel_single")

        return input_image
    
    @staticmethod
    def r_channel(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart r_channel")
        
        input_image = Dataset.rgb_channel_single(input_image, axis=0)
        
        if Dataset.verbose:
            print("-- End r_channel")

        return input_image, input_mask
    
    @staticmethod
    def rg_channel(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart rg_channel")
        
        r_channel = Dataset.rgb_channel_single(input_image, axis=0)[..., 0]
        g_channel = Dataset.rgb_channel_single(input_image, axis=1)[..., 0]
        input_image = tf.stack([r_channel, g_channel], axis=-1)
        
        if Dataset.verbose:
            print("-- End rg_channel")

        return input_image, input_mask
    
    @staticmethod
    def pad_channel_to_3(input_image):
        if Dataset.verbose:
            print("- Sart pad_channel_to_3")
        
        c = list(input_image.shape)[-1]
        empty = tf.zeros_like(input_image[..., 0])
        cs = tf.unstack(input_image, axis=-1)
        
        if c == 1:
            input_image = tf.stack([cs[0], cs[0], cs[0]], axis=-1)
        elif c == 2:
            input_image = tf.stack([cs[0], cs[1], empty], axis=-1)
        
        if Dataset.verbose:
            print("-- End pad_channel_to_3")

        return input_image
    
    @staticmethod
    def pad_input_channel_to_3(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart pad_input_channel_to_3")
        
        input_image = Dataset.pad_channel_to_3(input_image)
        
        if Dataset.verbose:
            print("-- End pad_input_channel_to_3")

        return input_image, input_mask
    
    @staticmethod
    def random_patches(input_image, input_mask, num_patches=100, patch_size=16):
        if Dataset.verbose:
            print("- Sart random_patches")
            
        h, w, c1 = input_image.shape
        h, w, c2 = input_mask.shape
    
        input_image_channels = tf.unstack(input_image, axis=-1)
        input_mask_channels = tf.unstack(input_mask, axis=-1)

        image_with_mask = tf.stack(input_image_channels + input_mask_channels, axis=-1)

        input_image_channels = None
        input_mask_channels = None

        """Get `num_patches` random crops from the image"""
        patches = []
        for i in range(num_patches):
            patch = tf.image.random_crop(image_with_mask, [patch_size, patch_size, c1+c2])
            patches.append(patch)

        patches = tf.stack(patches)
        channels = tf.unstack(patches, axis=-1)
        input_image = tf.stack(channels[:c1], axis=-1)
        input_mask = tf.stack(channels[c1:], axis=-1)

        channels = None
        if Dataset.verbose:
            print("-- End random_patches")

        return input_image, input_mask
    
    @staticmethod
    def get_patches_single(input_image, crop_size=(1000, 1000)):
        if Dataset.verbose:
            print("- Sart get_patches_single")
        
        input_images = get_patches_single(input_image, crop_size)
        
        if Dataset.verbose:
            print("-- End get_patches_single")

        return input_images
    
    @staticmethod
    def get_patches(input_image, input_mask, crop_size=(16, 16)):
        if Dataset.verbose:
            print("- Sart get_patches")
        
        input_images = Dataset.get_patches_single(input_image, crop_size)
        input_masks = Dataset.get_patches_single(input_mask, crop_size)
        
        if Dataset.verbose:
            print("-- End get_patches")

        return input_images, input_masks
    
    @staticmethod
    def get_over_patches_single(input_image, crop_size=(1000, 1000)):
        if Dataset.verbose:
            print("- Sart get_over_patches_single")
        
        input_images = get_over_patches_single(input_image, crop_size)
        
        if Dataset.verbose:
            print("-- End get_over_patches_single")

        return input_images
    
    @staticmethod
    def get_over_patches(input_image, input_mask, crop_size=(16, 16)):
        if Dataset.verbose:
            print("- Sart get_over_patches")
        
        input_images = Dataset.get_over_patches_single(input_image, crop_size)
        input_masks = Dataset.get_over_patches_single(input_mask, crop_size)
        
        if Dataset.verbose:
            print("-- End get_over_patches")

        return input_images, input_masks

    @staticmethod
    def resize_image_single(input_image, size=1024):
        if Dataset.verbose:
            print("- Sart resize_image_single")
        
        input_image = tf.cast(tf.image.resize(input_image, (size, size)), input_image.dtype)
        
        if Dataset.verbose:
            print("-- End resize_image_single")

        return input_image

    @staticmethod
    def resize_image(input_image, input_mask, size=1024):
        if Dataset.verbose:
            print("- Sart resize_image")
        
        input_image = Dataset.resize_image_single(input_image, size)
        input_mask = Dataset.resize_image_single(input_mask, size)
        
        if Dataset.verbose:
            print("-- End resize_image")

        return input_image, input_mask

    @staticmethod
    def resize_image_rectangle_single(input_image, crop_size=(1024, 1024)):
        if Dataset.verbose:
            print("- Sart resize_image_rectangle_single")
        
        input_image = tf.cast(tf.image.resize(input_image, crop_size), input_image.dtype)
        
        if Dataset.verbose:
            print("-- End resize_image_rectangle_single")

        return input_image

    @staticmethod
    def resize_rectangle_image(input_image, input_mask, crop_size=(1024, 1024)):
        if Dataset.verbose:
            print("- Sart resize_rectangle_image")
        
        input_image = Dataset.resize_image_rectangle_single(input_image, crop_size)
        input_mask = Dataset.resize_image_rectangle_single(input_mask, crop_size)
        
        if Dataset.verbose:
            print("-- End resize_rectangle_image")

        return input_image, input_mask

    @staticmethod
    def get_ratio(mask):
        if Dataset.verbose:
            print("- Sart get_ratio")
        image = mask
        ratio = tf.math.reduce_sum(image) / image.shape[0] / image.shape[1]
        if Dataset.verbose:
            print("-- End get_ratio")

        return ratio
    
    @staticmethod
    def to_categorical(input_image, input_mask):
        if Dataset.verbose:
            print("- Sart to_categorical")
        input_mask = tf.cast(input_mask, bool)
        false_mask = (input_mask == False)[..., 0]
        true_mask = (input_mask == True)[..., 0]
        input_mask = tf.cast(tf.stack([false_mask, true_mask], axis=-1), tf.float32)
        if Dataset.verbose:
            print("-- End to_categorical")

        return input_image, input_mask

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.seed = seed
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.image.random_flip_left_right
        self.augment_labels = tf.image.random_flip_left_right

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs, seed=self.seed)
        labels = self.augment_labels(labels, seed=self.seed)
        return inputs, labels

def display(
        display_list,
        show = True,
        figsize = (15, 15),
        dpi = 100,
        title = ['Input Image', 'True Mask', 'Predicted Mask'],
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for i in range(len(display_list)):
        gray_args = {}
        
        if i == 0:
            img = display_list[i]
            channel = list(img.shape)[-1]
            if channel == 1:
                display_list[i] = display_list[i][..., 0]
                gray_args = {
                    'cmap': 'gray',
                    'vmin': 0,
                    'vmax': 1,
                }
        else:
            if display_list[i].shape[-1] == 1:
                display_list[i] = display_list[i][..., 0]
            if display_list[i].shape[-1] == 2:
                display_list[i] = np.argmax(display_list[i], axis=-1)
                
        
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i], **gray_args)
        plt.axis('off')
    if show:
        plt.show()
    else:
        return fig