/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_PARSER_H
#define NPU_DEVICE_CORE_NPU_PARSER_H

#include <utility>

#include "npu_types.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

#include "graph/types.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"

namespace npu {
const std::string kInputDesc = "input_tensor_desc";
const std::string kOutputDesc = "output_tensor_desc";
const std::string kFormat = "serialize_format";
const std::string kType = "serialize_datatype";
const std::string kShape = "serialize_shape";

template <typename T>
static inline tensorflow::AttrValue BuildDescAttr(T shapes, TensorDataTypes types) {
  tensorflow::AttrValue desc_attr;
  for (size_t i = 0; i < types.size(); i++) {
    auto desc = desc_attr.mutable_list()->add_func();
    desc->set_name(std::to_string(i));

    tensorflow::AttrValue shape_value;
    if (shapes[i].unknown_rank()) {
      const int kUnknownRankDimSize = -2;
      shape_value.mutable_list()->add_i(kUnknownRankDimSize);
    } else {
      for (int j = 0; j < shapes[i].dims(); j++) {
        shape_value.mutable_list()->add_i(shapes[i].dim_size(j));
      }
    }
    (void)desc->mutable_attr()->insert({kShape, shape_value});

    tensorflow::AttrValue type_value;
    type_value.set_i(static_cast<int64_t>(types[i]));
    (void)desc->mutable_attr()->insert({kType, type_value});

    tensorflow::AttrValue format_value;
    format_value.set_i(static_cast<int>(ge::Format::FORMAT_NHWC));
    (void)desc->mutable_attr()->insert({kFormat, format_value});
  }
  return desc_attr;
}

/**
 * @breif: assemble desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorPartialShapes shapes, TensorDataTypes types, const std::string &name,
                  tensorflow::NodeDef &ndef);

/**
 * @breif: assemble desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorShapes shapes, TensorDataTypes types, const std::string &name, tensorflow::NodeDef &ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types,
                                           tensorflow::NodeDef &ndef);

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types,
                                            tensorflow::NodeDef &ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef);

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node &n);

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node &n);

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n);

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n);

/**
 * @breif: assemble op def
 * @param op_data: tensorflow op registration data
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOpDef(const tensorflow::OpRegistrationData &op_data, tensorflow::Node &n);

/**
 * @breif: assemble op def
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOpDef(tensorflow::Node &n);

/**
 * @breif: assemble op def
 * @param op_data: tensorflow op registration data
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED inline void AssembleOpDef(const tensorflow::OpRegistrationData &op_data,
                                              tensorflow::NodeDef &ndef) {
  std::string serialized_op_def;
  (void)op_data.op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, &ndef);
}

/**
 * @breif: assemble op def
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED inline void AssembleOpDef(tensorflow::NodeDef &ndef) {
  const tensorflow::OpRegistrationData *op_reg_data;
  (void)tensorflow::OpRegistry::Global()->LookUp(ndef.op(), &op_reg_data);
  std::string serialized_op_def;
  (void)op_reg_data->op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, &ndef);
}

void AssembleParserAddons(TFE_Context *context, tensorflow::Graph *graph);

void AssembleParserAddons(const tensorflow::FunctionLibraryDefinition *lib_def, tensorflow::Graph *graph);
}  // namespace npu
#endif  // NPU_DEVICE_CORE_NPU_PARSER_H
