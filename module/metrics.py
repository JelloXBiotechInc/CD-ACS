import tensorflow as tf
from keras import backend as K

BATCH_SIZE = 1
OUTPUT_CLASSES = 1

def jacard_coef_metrics(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true) / OUTPUT_CLASSES
    y_pred_f = K.flatten(y_pred) / OUTPUT_CLASSES
    intersection = K.sum(y_true_f * y_pred_f)
    jacard = (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    return jacard

def dice_coef_metrics(y_true, y_pred, target_value=1):
    y_true = tf.cast(y_true==target_value, tf.int32)
    y_pred = tf.cast(y_pred==target_value, tf.int32)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    TP = K.sum(y_true * y_pred) + 1
    P = K.sum(y_true) + 1
    PP = K.sum(y_pred) + 1
    
    Recall = TP / P
    Precision = TP / PP
    dice = 2 * Precision * Recall / (Precision + Recall)
    return dice

def batch_iter(y_trues, y_preds):
    for i in range(BATCH_SIZE):
        y_true = y_trues[i, ...]
        y_pred = y_preds[i, ...]
        yield y_true, y_pred
        
def batch_mean_result(iterator, preprocess, metric):
    results = []
    for y_true, y_pred in iterator:
        y_true, y_pred = preprocess(y_true, y_pred)
        results.append(metric(y_true, y_pred))
    return K.mean(tf.convert_to_tensor(results))

def jacard_coef(y_trues, y_preds):
    return batch_mean_result(
        iterator=batch_iter(y_trues, y_preds),
        preprocess=lambda y_true, y_pred: (y_true, tf.argmax(y_pred, axis=-1)),
        metric=jacard_coef_metrics,
    )

def jacard_coef_loss(y_trues, y_preds):
    return 1 - jacard_coef(y_trues, y_preds)

def dice_coef(y_trues, y_preds):
    return batch_mean_result(
        iterator=batch_iter(y_trues, y_preds),
        preprocess=lambda y_true, y_pred: (y_true, tf.argmax(y_pred, axis=-1)),
        metric=dice_coef_metrics,
    )

def dice_coef_loss(y_trues, y_preds):
    return 1 - dice_coef(y_trues, y_preds)

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import gc
import cv2
from multiprocessing import Pool, Process

def load_img_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = thresh // 255
    
    return img

def load_img(image_path):
    img = cv2.imread(image_path)[:,:,::-1]
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img

#this is the function to be parallelised
def multi_load_img(image_path):
    if image_path == '':
        img = None
    else:
        img = load_img_grayscale(image_path)
    return img

def cal_metrics_for_memory_imgs(gt, seg, file_name=''):
    tp = np.logical_and(gt==1, seg==1) # TP
    fp = np.logical_and(gt!=1, seg==1) # FP
    fn = np.logical_and(gt==1, seg!=1) # FN
    tn = np.logical_and(gt!=1, seg!=1) # TN

    TP = np.sum(tp)
    FP = np.sum(fp)
    FN = np.sum(fn)
    TN = np.sum(tn)

    intersection = tp

    iou_score = (np.sum(intersection) + 1) / ((np.sum(seg) + np.sum(gt) - np.sum(intersection)) + 1)

    dice = (np.sum(intersection)*2.0 + 1) / ((np.sum(seg) + np.sum(gt)) + 1)

    accuracy = (TP+TN+1)/(TP+TN+FP+FN+1)
    precision = (TP + 1) / (TP + FP + 1)
    recall = (TP + 1) / (TP + FN + 1)
    specificity = (TN + 1) / (TN + FP + 1)
    f1 = 2 / (1/precision + 1/recall)

    mts = {
        'file_name': file_name,
        'dice': dice,
        'jaccard': iou_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
    }
    return mts

def multi_cal_mts(files):
    if len(files) == 2:
        gt_file, pred_file = files
        rename_file = os.path.basename(gt_file).split('.')[0]
    elif len(files) == 3:
        gt_file, pred_file, rename_file = files
    else:
        raise ValueError(f"Object : {files} has length of : {len(files)}")
    
    pred_mask = multi_load_img(pred_file)
    height, width = pred_mask.shape

    mask = multi_load_img(gt_file)[:height, :width]
    mts = cal_metrics_for_memory_imgs(mask, pred_mask, rename_file)

    gc.collect()
    return mts

class _multi_load_imgs_cls:
    def __init__(self, callback=None):
        self.callback = callback
    
    def __call__(self, file):
        img = load_img(file)
        
        if self.callback != None:
            img = self.callback(img)
        
        return img

def multi_load_imgs(files, callback=None):
    imgs = []
    _multi_load_imgs = _multi_load_imgs_cls(callback)
    with Pool(8) as process:
        imgs = list(tqdm(process.imap(_multi_load_imgs, files), total=len(files)))
    return imgs

def cal_metrics(compare_files):
    metrics = []
    with Pool(len(compare_files)) as process:
        metrics = list(tqdm(process.imap(multi_cal_mts, compare_files), total=len(compare_files)))
    return metrics

def format_metrics(metrics):
    df = pd.DataFrame(metrics, columns=metrics[0].keys())
    df = df.set_index('file_name')
    return df

def rstrip_substring(m_str, substr):
    if m_str.endswith(substr):
        return m_str[:-len(substr)]
    return m_str