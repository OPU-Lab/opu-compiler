/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef SRC_RELAY_PASS_QUANTIZE_H_
#define SRC_RELAY_PASS_QUANTIZE_H_
/* 
 * quantize weights/bias/intermediate feature maps results
 * weights/bias: traverse data flow graph, quantize constant
 * intermediate feature maps: feed with user-defined input tensor,
 * deploy and compute each layer, then quantize each layer input 
 *
 * example for layer execution following topological order:  
 * 0 as input tensor, 1,2,3 as opu layers 
 *         0
 *       /  \
 *      1   2  : 0->1->2->3
 *      \  /
 *       3
 * quantize: 0(input for 1, 2), (1,2)(together, input for 3)
 */

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <cfloat>
#include <unordered_map>
#include <vector>

#include "./pattern_util.h"
#include "./hw_info.h"
#include "./gen_hw_ir.h"

#define FL_SEARCH_MAX 25

namespace tvm {
namespace relay {
class QNN : private ExprVisitor {
 public:
  // Map: opnode - fn_node
  std::unordered_map<const tvm::Object*, Func*> fmap_;
  std::vector<Func*> funcs;

  // Traverse dfg and quantize constants(e.g. weight/bias) and fm
  void Prepare(const Expr& body);
  
  // Driver function for qnorm
  void Normalize(const Expr& e);
  
  // Deloyment utilities
  DLContext ctx {kDLCPU, 0};
  DLDataType dtype {kDLFloat, 32, 1};
  tvm::runtime::NDArray Compute(
    const Expr& body,
    std::vector<tvm::runtime::NDArray> input_ndarrays);
    
  // top function -- used to identify input tensor shape
  const FunctionNode* fn_top {nullptr};
  std::vector<int64_t> input_shape;

  // flag to write weight/bias to disk as .npy
  bool dump_ {false};
  // sub-flag to write unquantized constant to disk
  bool dump_unquantized_constant_ {false};

 private:
  std::ostringstream os;

  void VisitExpr_(const FunctionNode* fn);

  void VisitExpr_(const CallNode* call);

  void VisitExpr_(const ConstantNode* op);

  // Quantization utilities
  // Map: layer_id - ndarray (layer output tensor)
  std::unordered_map<size_t, tvm::runtime::NDArray> output_dict;
  // Get tensor size
  int64_t GetTensorSize(runtime::NDArray data);
  // Search fraclen wrappers
  int SearchFraclenUtil(float* dl, size_t size, int wordlen);
  int SearchArrayFraclen(const std::vector<tvm::runtime::NDArray>& ndarrays);
  int SearchFraclen(tvm::runtime::NDArray ndarray);
  // Quantize data with designated wordlen and fraclen (signed fix-point format)
  float Convert(float* data, size_t size, int wordlen, int fraclen,
    bool use_round, float* data_o);
  // tvm::Relay::Type -> std::vector
  std::vector<int64_t> TypeToShape(Type ty);
  // Alter weight layout
  std::vector<float> AlterLayout(std::vector<float> src, std::vector<size_t>& shape,
                            std::string layout, std::string target_layout = "HWIO");
  // input fraction length dependency map (for residual add case)
  std::unordered_map<int64_t, int64_t> ifl_dep_index_map_;
  void CheckQuantizeConstraint();
  class QCC;
  class QNorm;
};

class QNN::QCC {
 public:
  int* parent;
  size_t size;
  std::unordered_map<int, std::vector<int>> gmap_;
  std::unordered_map<int, int> vmap_;
  std::ostringstream os;
  
  QCC(size_t size) {
    this->size = size*2;
    parent = new int[this->size];
    std::memset(parent, -1, sizeof(int)*this->size);
  }
  int Find(int index);
  void Union(int x, int y);
  int GetOutput(int layer_index);
  int GetInput(int layer_index);
  void Prepare(std::vector<Func*> funcs);
  void Dump();
  void SetInput(int layer_index, int value);
  void SetOutput(int layer_index, int value);
  void Apply(std::vector<Func*> funcs);
};

class QNN::QNorm : private ExprMutator {
  public:
    std::ostringstream os;

    std::unordered_map<const tvm::Object*, Func*> fmap_;
    std::unordered_map<size_t, float> scale_map_;
    std::unordered_map<size_t, float> scale_map_b_;
    std::unordered_map<size_t, bool> scale_;
    std::unordered_map<size_t, Func*> id_func_map_;
    
    void SetFmap(std::unordered_map<const tvm::Object*, Func*>& fmap);
    void SetScaleMap(std::unordered_map<size_t, tvm::runtime::NDArray>& output_dict);
    
    Expr Transform(const Expr& body);

    Expr VisitExpr_(const ConstantNode* op);
    std::vector<float> AlterLayout(std::vector<float> src, std::vector<size_t>& shape,
                               std::string layout, std::string target_layout = "HWIO");
};

}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_QUANTIZE_H_
