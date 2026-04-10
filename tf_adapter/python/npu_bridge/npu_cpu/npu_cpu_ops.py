#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
import tensorflow.compat.v1 as tf
from npu_bridge.helper import helper

gen_npu_cpu_ops = helper.get_gen_ops()


## 提供device侧FeatureMapping LookupOrInsert功能
#  @param table_handle int64 类型
#  @param keys int64 类型
#  @param bucket_size int 类型
#  @param embedding_dim int 类型
#  @param filter_mode string 类型
#  @param filter_freq int 类型
#  @param default_key_or_value bool 类型
#  @param default_key int 类型
#  @param default_value float 类型
#  @param filter_key_flag bool 类型
#  @param filter_key int 类型
#  @return values float 类型
def embedding_hashtable_lookup_or_insert(table_handle, keys, bucket_size, embedding_dim, filter_mode, filter_freq,
                                         default_key_or_value, default_key, default_value, filter_key_flag, filter_key):
    """ device embedding feature mapping lookup or insert. """
    result = gen_npu_cpu_ops.EmbeddingHashTableLookupOrInsert(
        table_handle=table_handle, keys=keys, bucket_size=bucket_size, embedding_dim=embedding_dim,
        filter_mode=filter_mode, filter_freq=filter_freq, default_key_or_value=default_key_or_value,
        default_key=default_key, default_value=default_value, filter_key_flag=filter_key_flag, filter_key=filter_key)
    return result


## 提供embeddingrankid功能
#  @param addr_tensor tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param index tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param row_memory int类型，一行数据存储的大小 默认为320。
#  @param mode string类型，embeddingrankid的操作类型，可以为”mod”,”order”;数据存储的方式。
#  @return 对输入addr_tensor，index_tensor执行完embeddingrankid操作之后的结果tensor
def embeddingrankid(addr_tensor, index, row_memory=320, mode='mod'):
    """ Embed rank index. """
    result = gen_npu_cpu_ops.embedding_rank_id(
        addr_table=addr_tensor,
        index=index,
        row_memory=row_memory,
        mode=mode)
    return result


## 提供embeddinglocalindex功能
#  @param addr_tensor tensorflow的tensor类型，embeddinglocalindex操作的输入；
#  @param index tensorflow的tensor类型，embeddinglocalindex操作的输入；
#  @param row_memory int类型，一行数据存储的大小 默认为320。
#  @param mode string类型，embeddinglocalindex的操作类型，可以为”mod”,”order”;数据存储的方式。
#  @return 对输入addr_tensor，index_tensor执行完embeddinglocalindex操作之后的结果tensor
def embedding_local_index(addr_tensor, index, row_memory=320, mode='mod'):
    """ Embed local index. """
    result = gen_npu_cpu_ops.embedding_local_index(
        addr_table=addr_tensor,
        index=index,
        row_memory=row_memory,
        mode=mode)
    return result


## 提供RandomChoiceWithMask功能
#  @param x bool 类型
#  @param count int 类型
#  @param seed int类型
#  @param seed2 int类型
#  @return y int32类型 mask bool 类型
def randomchoicewithmask(x, count, seed=0, seed2=0):
    """ Random choice with mask. """
    result = gen_npu_cpu_ops.random_choice_with_mask(
        x=x,
        count=count,
        seed=seed,
        seed2=seed2)
    return result


## 提供DenseImageWarp功能
#  @param image tensor类型
#  @param flow tensor类型
#  @return y tensor类型
def dense_image_warp(image, flow, name=None):
    """ Dense image warp. """
    result = gen_npu_cpu_ops.dense_image_warp(
        image=image,
        flow=flow,
        name=name
    )
    return result


## 提供host侧StatelessDropout功能
#  @param x float32,float16,bfloat16 类型
#  @param noise_shape int64 类型
#  @param p float32,float16,bfloat16 类型
#  @param seed int64 类型
#  @param offset int64 类型
#  @return values float32,float16,bfloat16 类型
def stateless_dropout(x, noise_shape, p, seed, offset):
    """ host stateless_dropout. """
    result = gen_npu_cpu_ops.StatelessDropout(
        x=x,
        noise_shape=noise_shape,
        p=p,
        seed=seed,
        offset=offset
    )
    return result


## DenseImageWarp的梯度函数
@ops.RegisterGradient("DenseImageWarp")
def dense_image_warp_grad(op, grad):
    """ Dense image warp grad. """
    image = op.inputs[0]
    flow = op.inputs[1]
    grad_image, grad_flow = gen_npu_cpu_ops.dense_image_warp_grad(
        grad, image, flow)
    return [grad_image, grad_flow]


## 提供BatchEnqueue功能
#  @param x uint8 类型
#  @param queue_id uint32 类型
#  @param batch_size int 类型
#  @param queue_name string 类型
#  @param queue_depth int64 类型
#  @param pad_mode string 类型
#  @return enqueue_count int64类型
def batch_enqueue(x, queue_id=0, batch_size=8, queue_name="", queue_depth=100, pad_mode="REPLICATE"):
    """ Batch enqueue. """
    result = gen_npu_cpu_ops.batch_enqueue(
        x=x,
        queue_id=queue_id,
        batch_size=batch_size,
        queue_name=queue_name,
        queue_depth=queue_depth,
        pad_mode=pad_mode)
    return result


## 提供OCRRecognitionPreHandle功能
#  @param imgs_data uint8 类型
#  @param imgs_offset int32 类型
#  @param imgs_size int32 类型
#  @param langs int32 类型
#  @param langs_score int32 类型
#  @param batch_size int 类型
#  @param data_format string 类型
#  @param pad_mode string 类型
#  @return imgs,imgs_relation,imgs_lang,imgs_piece_fillers uint8,int32,int32,int32 类型
def ocr_recognition_pre_handle(imgs_data, imgs_offset, imgs_size, langs, langs_score, \
                               batch_size=8, data_format="NHWC", pad_mode="REPLICATE"):
    """ Recognize ocr pre-handle. """
    result = gen_npu_cpu_ops.ocr_recognition_pre_handle(
        imgs_data=imgs_data,
        imgs_offset=imgs_offset,
        imgs_size=imgs_size,
        langs=langs,
        langs_score=langs_score,
        batch_size=batch_size,
        data_format=data_format,
        pad_mode=pad_mode)
    return result


## 提供OCRDetectionPreHandle功能
#  @param img uint8 类型
#  @param data_format string 类型
#  @return resized_img,h_scale,w_scale uint8,float32,float32 类型
def ocr_detection_pre_handle(img, data_format="NHWC"):
    """
    ocr detection pre-handle
    """
    result = gen_npu_cpu_ops.ocr_detection_pre_handle(
        img=img,
        data_format=data_format)
    return result


## 提供OCRIdentifyPreHandle功能
#  @param imgs_data uint8 类型
#  @param imgs_offset int32 类型
#  @param imgs_size int32 类型
#  @param size list(int) 类型
#  @param data_format string 类型
#  @return resized_imgs, uint8 类型
def ocr_identify_pre_handle(imgs_data, imgs_offset, imgs_size, size, data_format="NHWC"):
    """ Ocr identification pre-handle. """
    result = gen_npu_cpu_ops.ocr_identify_pre_handle(
        imgs_data=imgs_data,
        imgs_offset=imgs_offset,
        imgs_size=imgs_size,
        size=size,
        data_format=data_format)
    return result


## 提供BatchDilatePolys功能
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param score float 类型
#  @param min_border int32 类型
#  @param min_area_thr int32 类型
#  @param score_thr float 类型
#  @param expand_scale float 类型
#  @return dilated_polys_data int32 类型
#  @return dilated_polys_offset int32 类型
#  @return dilated_polys_size int32 类型
def batch_dilate_polys(polys_data, polys_offset, polys_size, score, \
                       min_border, min_area_thr, score_thr, expand_scale):
    """ Batch dilate poly. """
    result = gen_npu_cpu_ops.batch_dilate_polys(
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        score=score,
        min_border=min_border,
        min_area_thr=min_area_thr,
        score_thr=score_thr,
        expand_scale=expand_scale)
    return result


## 提供OCRFindContours功能
#  @param img uint8 类型
#  @param value_mode int 类型
#  @return polys_data int32 类型
#  @return polys_offset int32 类型
#  @return polys_size int32 类型
def ocr_find_contours(img, value_mode=0):
    """ Ocr find contours. """
    result = gen_npu_cpu_ops.ocr_find_contours(img=img, value_mode=value_mode)
    return result


## 提供Dequeue功能
#  @param queue_id uint32 类型
#  @param output_type RealNumberType 类型
#  @param output_shape list(int) 类型
#  @param queue_name string 类型
#  @return data 根据output_type确定类型
def dequeue(queue_id, output_type, output_shape, queue_name=""):
    """ Dequeue. """
    result = gen_npu_cpu_ops.dequeue(
        queue_id=queue_id,
        output_type=output_type,
        output_shape=output_shape,
        queue_name=queue_name)
    return result


## 提供OCRDetectionPostHandle功能
#  @param img uint8 类型
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param data_format string 类型
#  @return imgs_data,imgs_offset,imgs_size,rect_points uint8,int32,int32,int32 类型
def ocr_detection_post_handle(img, polys_data, polys_offset, polys_size, data_format="NHWC"):
    """ Orc detection post-handle. """
    result = gen_npu_cpu_ops.ocr_detection_post_handle(
        img=img,
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        data_format=data_format)
    return result


## 提供WarpAffineV2功能
#  @param x uint8, float32 类型
#  @param matrix float32 类型
#  @param dst_size int32, int64 类型
#  @param interploation string 类型
#  @param border_type string 类型
#  @param border_value int 类型
#  @return y uint8, float32 类型
def warp_affine_v2(x, matrix, dst_size, interploation="INTEL_BILINEAR", border_type="BORDER_CONSTANT", border_value=0):
    """ Warp Affine V2. """
    result = gen_npu_cpu_ops.warp_affine_v2(
        x=x,
        matrix=matrix,
        dst_size=dst_size,
        interploation=interploation,
        border_type=border_type,
        border_value=border_value)
    return result


## 提供ResizeV2功能
#  @param x uint8, float32 类型
#  @param dst_size int32, int64 类型
#  @param interploation string 类型
#  @return y uint8, float32 类型
def resize_v2(x, dst_size, interploation="INTEL_BILINEAR"):
    """ Resize V2. """
    result = gen_npu_cpu_ops.resize_v2(
        x=x,
        dst_size=dst_size,
        interploation=interploation)
    return result


## 提供ResizeAndClipPolys功能
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param h_scale float32 类型
#  @param w_scale float32 类型
#  @param img_h int32 类型
#  @param img_w int32 类型
#  @return clipped_polys_data,clipped_polys_offset,clipped_polys_size int32,int32,int32 类型
def resize_and_clip_polys(polys_data, polys_offset, polys_size, h_scale, w_scale, img_h, img_w):
    """ Resize and clip polys. """
    result = gen_npu_cpu_ops.resize_and_clip_polys(
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        h_scale=h_scale,
        w_scale=w_scale,
        img_h=img_h,
        img_w=img_w)
    return result


## 提供NonZeroWithValueShape功能
#  @param value double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64, unit64, bool 类型
#  @param index int32 类型
#  @param count int32 类型
#  @return out_value,out_index double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64,
#                              unit64, bool,int32,int32 类型
def non_zero_with_value_shape(value, index, count):
    """ Non zero with value shape. """
    result = gen_npu_cpu_ops.non_zero_with_value_shape(
        value=value,
        index=index,
        count=count)
    return result


## 提供NonZeroWithValueShapeV2功能
#  @param value double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64, unit64, bool 类型
#  @param index int32 类型
#  @param count int32 类型
#  @return out_value,out_index double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64,
#                              unit64, bool,int32,int32 类型
def non_zero_with_value_shape_v2(value, index, count):
    """ Non zero with value shape. """
    result = gen_npu_cpu_ops.NonZeroWithValueShapeV2(
        value=value,
        index=index,
        count=count)
    return result


## 提供host侧FeatureMapping功能
#  @param feature_id int64 类型
#  @param threshold int 类型
#  @param table_name string 类型
#  @return offset_id int64 类型
def host_feature_mapping(feature_id, threshold=1, table_name="default_table_name"):
    """ host feature mapping. """
    result = gen_npu_cpu_ops.HostFeatureMapping(
        feature_id=feature_id,
        threshold=threshold,
        table_name=table_name)
    return result


## 提供device侧FeatureMapping功能
#  @param feature_id int64 类型
#  @return offset_id int32 类型
def device_feature_mapping(feature_id):
    """ device feature mapping. """
    result = gen_npu_cpu_ops.EmbeddingFeatureMapping(
        feature_id=feature_id)
    return result


## 提供device侧初始化hashmap表功能
#  @param table_id int32 类型
#  @param bucket_size int64 类型
#  @param load_factor int64 类型
#  @param embedding_dim int64 类型
#  @param dtype type 类型
#  @return table_handle int64 类型
def init_embedding_hashmap_v2(table_id, bucket_size, load_factor, embedding_dim, dtype):
    """ device init embedding hashmap v2. """
    result = gen_npu_cpu_ops.InitEmbeddingHashmapV2(
        table_id=table_id, bucket_size=bucket_size,
        load_factor=load_factor, embedding_dim=embedding_dim, dtype=dtype)
    return result


## 提供device侧去初始化hashmap表功能
#  @param table_id int32 类型
def deinit_embedding_hashmap_v2(table_id):
    """ device deinit embedding hashmap v2. """
    gen_npu_cpu_ops.DeinitEmbeddingHashmapV2(table_id=table_id)


## 提供device侧hashmap表映射功能
#  @param table_id int32 类型
#  @return table_handle int64 类型
def table_to_resource_v2(table_id, bucket_size, load_factor, embedding_dim, dtype):
    """ device embedding hashmap to handle. """
    result = gen_npu_cpu_ops.TableToResourceV2(table_id=table_id)
    return result


## 提供device侧计算hashmap表大小功能
#  @param table_ids int32 类型
#  @param filter_export_flag bool 类型
#  @param export_mode string 类型
#  @return table_sizes int64 类型
def embedding_hashmap_table_size_v2(table_ids, filter_export_flag, export_mode):
    """ device embedding hashmap table size. """
    result = gen_npu_cpu_ops.EmbeddingHashmapSize(
        table_ids=table_ids, filter_export_flag=filter_export_flag, export_mode=export_mode)
    return result


## 提供host侧hashmap导出功能
#  @param file_path string 类型
#  @param table_ids int32 类型
#  @param table_names string 类型
#  @param global_step int32/int64 类型
#  @param keys int64 类型
#  @param counters uint64 类型
#  @param filter_flag uint8 类型
#  @param values float32 类型
#  @param num int64 类型
def embedding_hashmap_export_v2(file_path, table_ids, table_names, global_step,
                                keys, counters, filter_flag, values, num):
    """ host embedding hashmap export. """
    gen_npu_cpu_ops.EmbeddingHashmapExport(
        file_path=file_path, table_ids=table_ids, table_names=table_names, global_step=global_step,
        keys=keys, counters=counters, filter_flag=filter_flag, values=values, num=num)


## 提供host侧hashmap文件大小功能
#  @param file_path string 类型
#  @param table_ids int32 类型
#  @param table_names string 类型
#  @param global_step int32/int64 类型
#  @param embedding_dims int64 类型
#  @return table_sizes int64 类型
def embedding_hashmap_file_size_v2(file_path, table_ids, table_names, global_step, embedding_dims):
    """ host embedding hashmap file size. """
    result = gen_npu_cpu_ops.EmbeddingHashmapFileSize(
        file_path=file_path, table_ids=table_ids, table_names=table_names,
        global_step=global_step, embedding_dims=embedding_dims)
    return result


## 提供host侧hashmap导入功能
#  @param file_path string 类型
#  @param table_ids int32 类型
#  @param table_sizes int64 类型
#  @param table_names string 类型
#  @param global_step int32/int64 类型
#  @param embedding_dims int 类型
#  @param num int64 类型
#  @return keys(int64)/counters(uint64)/filter_flag(uint8)/values(float32)
def embedding_hashmap_import_v2(file_path, table_ids, table_sizes, table_names, global_step, embedding_dims, num):
    """ host embedding feature mapping import. """
    result = gen_npu_cpu_ops.EmbeddingHashmapImport(
        file_path=file_path, table_ids=table_ids, table_sizes=table_sizes,
        table_names=table_names, global_step=global_step, embedding_dims=embedding_dims, num=num)
    return result


## EmbeddingHashTable Init功能
#  @param table_handle int64 类型
#  @param sampled_values float 类型
#  @param bucket_size int 类型
#  @param embedding_dim int 类型
#  @param initializer_mode string 类型
#  @param constant_value int 类型
def init_embedding_hashtable(table_handle, sampled_values, bucket_size, embedding_dim, initializer_mode,
                             constant_value):
    """ device init embedding hashtable. """
    result = gen_npu_cpu_ops.InitEmbeddingHashTable(
        table_handle=table_handle, sampled_values=sampled_values, bucket_size=bucket_size, embedding_dim=embedding_dim,
        initializer_mode=initializer_mode, constant_value=constant_value)
    return result


## 提供host侧hashTable导入功能
#  @param table_handles int64 类型
#  @param embedding_dims int64 类型
#  @param bucket_sizes int64 类型
#  @param keys int64 类型
#  @param counters uint64 类型
#  @param filter_flags uint8 类型
#  @param values float 类型
def embedding_hash_table_import(table_handles, embedding_dims, bucket_sizes, keys, counters, filter_flags, values):
    """ host embedding feature hash table import. """
    result = gen_npu_cpu_ops.EmbeddingHashTableImport(
        table_handles=table_handles, embedding_dims=embedding_dims, bucket_sizes=bucket_sizes,
        keys=keys, counters=counters, filter_flags=filter_flags, values=values)
    return result


## 提供host侧StatelessRandomChoiceWithMask功能
#  @param x bool 类型
#  @param count int32 类型
#  @param seed int64 类型
#  @param offset int64 类型
def stateless_random_choice_with_mask(x, count, seed, offset):
    """ host stateless random choice with mask. """
    result = gen_npu_cpu_ops.StatelessRandomChoiceWithMask(
        x=x, count=count, seed=seed, offset=offset)
    return result


## 提供host侧hashTable导出功能
#  @param table_handles int64 类型
#  @param table_sizes int64 类型
#  @param embedding_dims int64 类型
#  @param bucket_sizes int64 类型
#  @param export_mode string 类型
#  @param filtered_export_flag bool 类型
def embedding_hash_table_export(table_handles, table_sizes, embedding_dims, bucket_sizes, export_mode='all',
                                filter_export_flag=False):
    """ host embedding feature hash table export. """
    result = gen_npu_cpu_ops.EmbeddingHashTableExport(
        table_handles=table_handles, table_sizes=table_sizes, embedding_dims=embedding_dims, bucket_sizes=bucket_sizes,
        export_mode=export_mode, filter_export_flag=filter_export_flag)
    return result


## EmbeddingHashTableApplyAdamW AdamW 更新功能
#  @param table_handle int64 类型
#  @param keys int64 类型
#  @param m float16, float32 类型
#  @param v float16, float32 类型
#  @param beta1_power float16, float32 类型
#  @param beta2_power float16, float32 类型
#  @param lr float16, float32 类型
#  @param weight_decay float16, float32 类型
#  @param beta1 float16, float32 类型
#  @param beta2 float16, float32 类型
#  @param epsilon float16, float32 类型
#  @param grad float16, float32 类型
#  @param max_grad_norm float16, float32 类型
#  @param embedding_dim int 类型
#  @param bucket_size int 类型
#  @param amsgrad bool 类型
#  @param maximize bool 类型
def embedding_hashtable_apply_adam_w(table_handle, keys, m, v, beta1_power, beta2_power, lr, weight_decay,
                                     beta1, beta2, epsilon, grad, max_grad_norm, embedding_dim,
                                     bucket_size, amsgrad, maximize):
    """ device update embedding hashtable using AdamW. """
    result = gen_npu_cpu_ops.EmbeddingHashTableApplyAdamW(
        table_handle=table_handle, keys=keys, m=m, v=v, beta1_power=beta1_power, beta2_power=beta2_power,
        lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad,
        max_grad_norm=max_grad_norm, embedding_dim=embedding_dim, bucket_size=bucket_size,
        amsgrad=amsgrad, maximize=maximize)
    return result
