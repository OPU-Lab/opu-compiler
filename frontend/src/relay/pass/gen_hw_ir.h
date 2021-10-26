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
#ifndef SRC_RELAY_PASS_GEN_HW_IR_H_
#define SRC_RELAY_PASS_GEN_HW_IR_H_
/* 
 * extract operator parameters in each function according to OPU IR
 * (ir data structure in hw_info.h)
 */

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <stack>
#include <unordered_map>
#include <vector>

#include "./hw_info.h"

namespace tvm {
namespace relay {

struct Func{
  const tvm::Object* ref{nullptr};
  OpuInfo* info{nullptr};
  std::vector<Func*> input_funcs;
  std::vector<Func*> output_funcs;
};

class IRCollector : private ExprVisitor {
 public:
  Func* cfunc{nullptr};
  std::vector<Func*> funcs;
  // opnode - fn_node
  std::unordered_map<const tvm::Object*, Func*> fmap_;
  // fn_node - (arg)fn_node
  std::unordered_map<const tvm::Object*,
    std::vector<const tvm::Object*>> fn_arg_map_;
  // fn_node - Func
  std::unordered_map<const tvm::Object*, Func*> fn_func_map_;
  // model input shape
  Array<PrimExpr> top_input_shape;
  // Func - arg index of residual input Func
  std::unordered_map<Func*, std::vector<int>> func_res_idx_map_; 

  void MakeTopologicalIndex();
  void TopologicalSortUtil(
      Func* func,
      std::unordered_map<Func*, bool> &visited,
      std::stack<Func*> &Stack);

  void Prepare(const Expr& body);
  // opu ir syntax self-checking and correction for each layer itself
  void LocalCanonicalize();

  void WriteIR();

  bool use_post_padding {false};
 private:
  std::ostringstream os;

  bool EqFunc(const tvm::Object* e, Func* func);

  void Update(const tvm::Object* ref);

  void VisitExpr_(const FunctionNode* fn_node);

  void VisitExpr_(const TupleNode* op);

  void VisitExpr_(const CallNode* call);

  void VisitExpr_(const ConstantNode* op);

  const CallNode* HasConcat(const Expr& body);
  
  size_t FindFirstNonResidualInput(const FunctionNode* fn);
};

}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_GEN_HW_IR_H_
