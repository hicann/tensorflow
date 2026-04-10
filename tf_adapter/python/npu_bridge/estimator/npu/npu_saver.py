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

"""Definition for NPU saver"""

import tensorflow as tf
from tensorflow.python.training.saver import BulkSaverBuilder
from tensorflow.python.training.saver import Saver
from tensorflow.python.eager import context
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from npu_bridge.estimator.npu import util
from npu_bridge.hccl import hccl_ops


class NPUBulkSaverBuilder(BulkSaverBuilder):
    """Class to build NPU builker saver"""
    def _build_internal(self,
                        names_to_saveables,               # Dict[str, Variable/SaveableObject], mapping from names
                        reshape=False,                    # Allow restoring variables with different shapes
                        sharded=False,                    # Save checkpoints in sharded format (per device)
                        max_to_keep=5,                    # Max number of checkpoints to keep
                        keep_checkpoint_every_n_hours=10000.0,  # Interval (hours) to keep checkpoints permanently
                        name=None,                        # Optional op name prefix
                        restore_sequentially=False,       # Restore variables sequentially within each device
                        filename="model",                 # Base checkpoint filename
                        build_save=True,                  # Whether to build save ops
                        build_restore=True):              # Whether to build restore ops
        if not context.executing_eagerly() and (not build_save or not build_restore):
            raise ValueError("Save and restore operations must be built together "
                             "  when eager execution is disabled.")

        saveable_list = saveable_object_util.validate_and_slice_inputs(
            names_to_saveables)
        if max_to_keep is None:
            max_to_keep = 0

        with ops.name_scope(name, "save",
                            [saveable.op for saveable in saveable_list]) as name:
            # Create a placeholder string tensor for the checkpoint filename.
            file_name_tensor = array_ops.placeholder_with_default(
                filename or "model", shape=(), name="filename")

            # Preserve the name "Const" for backward compatibility.
            file_name_tensor = array_ops.placeholder_with_default(
                file_name_tensor, shape=(), name="Const")

            # Create the save operations.
            if sharded:
                per_device = self._GroupByDevices(saveable_list)
                if build_save:
                    op_list = []
                    with tf.name_scope("Save_Weight_Update_Sharding"):
                        grad_and_var_items = util.get_all_grad_item()
                        for item in grad_and_var_items:
                            if item.var in names_to_saveables:
                                rank_id = item.root_rank_id
                                if rank_id >= 0:
                                    with tf.get_default_graph().control_dependencies(op_list):
                                        out_var = hccl_ops.broadcast([item.var], rank_id, 2, rank_id)
                                    op_list.append(out_var[0].op)
                    if len(op_list) > 0:
                        with tf.get_default_graph().control_dependencies(op_list):
                            save_op_tensor = self._AddShardedSaveOps(file_name_tensor, per_device)
                    else:
                        save_op_tensor = self._AddShardedSaveOps(file_name_tensor, per_device)
                if build_restore:
                    restore_op_tensor = self._AddShardedRestoreOps(file_name_tensor, per_device,
                                                            restore_sequentially, reshape)
            else:
                if build_save:
                    op_list = []
                    with tf.name_scope("Save_Weight_Update_Sharding"):
                        grad_and_var_items = util.get_all_grad_item()
                        for item in grad_and_var_items:
                            if item.var in names_to_saveables:
                                rank_id = item.root_rank_id
                                if rank_id >= 0:
                                    with tf.get_default_graph().control_dependencies(op_list):
                                        out_var = hccl_ops.broadcast([item.var], rank_id, 2, rank_id)
                                    op_list.append(out_var[0].op)
                    if len(op_list) > 0:
                        with tf.get_default_graph().control_dependencies(op_list):
                            save_op_tensor = self._AddSaveOps(file_name_tensor, saveable_list)
                    else:
                        save_op_tensor = self._AddSaveOps(file_name_tensor, saveable_list)
                if build_restore:
                    restore_op_tensor = self._AddRestoreOps(file_name_tensor, saveable_list,
                                                     restore_sequentially, reshape)

        # In the following scenario, it's possible for restore_ops to be named
        # differently:
        # - Build an inference graph and export a meta_graph.
        # - Import the inference meta_graph.
        # - Extend the inference graph into a training graph.
        # - Export a new meta_graph.
        # Now the second restore_op_tensor may be called "restore_all_1".
        # Therefore, comment out the assert for now until we determine whether
        # supporting this usage pattern makes sense.
        #
        # assert restore_op_tensor.name.endswith("restore_all"), restore_op_tensor.name
        if context.executing_eagerly():
            # Store the tensor values to the tensor_names.
            save_tensor_name = save_op_tensor.numpy() if build_save else ""
            return saver_pb2.SaverDef(
                filename_tensor_name=file_name_tensor.numpy(),        # Name of the filename tensor
                save_tensor_name=save_tensor_name,                    # Tensor name used for saving
                restore_op_name="",                                   # Restore op name (unused in eager mode)
                max_to_keep=max_to_keep,                              # Maximum number of checkpoints to retain
                sharded=sharded,                                      # Whether checkpoints are sharded
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,  # Interval (hours) to keep checkpoints
                version=self._write_version)                          # SaverDef version
        default_graph = ops.get_default_graph()
        # Perform basic sanity checks on collections that contain
        # PartitionedVariables. If a saved collection includes a PartitionedVariable,
        # the GraphDef must contain concat ops to reconstruct the value (otherwise
        # a lookup error will occur when loading).
        check_collection_list = default_graph.get_all_collection_keys()
        for collection_type in check_collection_list:
            for element in default_graph.get_collection(collection_type):
                if isinstance(element, variables.PartitionedVariable):
                    try:
                        default_graph.get_operation_by_name(element.name)
                    except KeyError:
                        # Create a concat op for this PartitionedVariable. The user may
                        # not require it, but we'll attempt to look it up during MetaGraph restore
                        # since it appears in a collection.
                        element.as_tensor()
        return saver_pb2.SaverDef(
            filename_tensor_name=file_name_tensor.name,            # Name of the filename tensor
            save_tensor_name=save_op_tensor.name,                  # Save op tensor name
            restore_op_name=restore_op_tensor.name,                # Restore op name
            max_to_keep=max_to_keep,                               # Maximum number of checkpoints to keep
            sharded=sharded,                                       # Whether checkpoints are sharded
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,  # Interval (hours) to keep checkpoints
            version=self._write_version)                           # SaverDef version


class NPUSaver(Saver):
    """NPU saver for saving checkpoints"""

    class NPUSaver(Saver):
        """NPU saver for saving checkpoints"""

    def __init__(self,
                 var_list=None,                      # Variables or SaveableObjects to save
                 reshape=False,                      # Allow restoring variables with different shapes
                 sharded=False,                      # Save checkpoints in sharded format
                 max_to_keep=5,                      # Maximum number of checkpoints to retain
                 keep_checkpoint_every_n_hours=10000.0,  # Interval (hours) to keep checkpoints permanently
                 name=None,                          # Optional saver name scope
                 restore_sequentially=False,         # Restore variables sequentially per device
                 saver_def=None,                     # Existing SaverDef proto (if provided)
                 builder=None,                       # Custom SaverBuilder instance
                 defer_build=False,                  # Delay building save/restore ops
                 allow_empty=False,                  # Allow empty var_list
                 write_version=saver_pb2.SaverDef.V2,  # Checkpoint format version
                 pad_step_number=False,              # Pad global step number in filenames
                 save_relative_paths=False,          # Save checkpoint paths as relative
                 filename=None):                     # Base checkpoint filename
        super(NPUSaver, self).__init__(
            var_list=var_list,
            reshape=reshape,                         # Allow restoring variables with different shapes
            sharded=sharded,                         # Save checkpoints in sharded format
            max_to_keep=max_to_keep,                 # Maximum number of checkpoints to retain
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,  # Interval (hours) to keep checkpoints
            name=name,                               # Optional saver name scope
            restore_sequentially=restore_sequentially,  # Restore variables sequentially per device
            saver_def=saver_def,
            builder=builder,
            defer_build=defer_build,
            allow_empty=allow_empty,
            write_version=write_version,
            pad_step_number=pad_step_number,
            save_relative_paths=save_relative_paths,
            filename=filename
        )
        self._builder = None  # Internal builder used for constructing NPU saver ops

    def _build(self, checkpoint_path, build_save, build_restore):
        if not self.saver_def or context.executing_eagerly():
            if self._builder is None:
                self._builder = NPUBulkSaverBuilder(self._write_version)
        super()._build(checkpoint_path=checkpoint_path, build_save=build_save, build_restore=build_restore)
