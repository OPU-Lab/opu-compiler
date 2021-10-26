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

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/ndarray.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class AlterToNCHWLayout : public ExprMutator {
 public:
  Expr VisitExpr_(const VarNode* var) final {
    const auto* rtype = var->checked_type().as<TensorTypeNode>();
    std::vector<IndexExpr> input_shape = {
      rtype->shape[0],
      rtype->shape[3],
      rtype->shape[1],
      rtype->shape[2]
    };
    auto var_type = TensorTypeNode::make(input_shape, DataType::Float(32));
    return VarNode::make("input", var_type);   
  }
  
  Expr VisitExpr_(const CallNode* call) final {  
    std::cout << call->op << "\n";
    if (call->op == Op::Get("nn.conv2d")) {
      Expr ifm;
      if (call->args[0].get()->IsInstance<CallNode>()) {
        ifm = VisitExpr_(call->args[0].as<CallNode>());
      } else if (call->args[0].get()->IsInstance<VarNode>()) {
        ifm = VisitExpr_(call->args[0].as<VarNode>());
      }
      Expr weight = VisitExpr_(call->args[1].as<ConstantNode>());
      auto a = call->attrs.as<Conv2DAttrs>();
      std::string data_layout = "NCHW";
      std::string kernel_layout = "OIHW";
      return Conv2D(ifm, weight, a->strides, a->padding, a->dilation, a->groups,
                    a->channels, a->kernel_size, data_layout, kernel_layout,
                    data_layout, a->out_dtype);  
    } else if (call->op == Op::Get("nn.pad")) {
      Expr ifm;
      if (call->args[0].get()->IsInstance<CallNode>()) {
        ifm = VisitExpr_(call->args[0].as<CallNode>());
      } else if (call->args[0].get()->IsInstance<VarNode>()) {
        ifm = VisitExpr_(call->args[0].as<VarNode>());
      }
      auto a = call->attrs.as<PadAttrs>();
      Array<Array<IndexExpr>> pad_width = {
        {a->pad_width[0][0], a->pad_width[0][1]},
        {a->pad_width[3][0], a->pad_width[3][1]},
        {a->pad_width[1][0], a->pad_width[1][1]},
        {a->pad_width[2][0], a->pad_width[2][1]}
      };
      return Pad(ifm, pad_width, a->pad_value, a->pad_mode);
    } else if (call->op == Op::Get("nn.max_pool2d")) {
      Expr ifm = VisitExpr_(call->args[0].as<CallNode>());
      auto a = call->attrs.as<MaxPool2DAttrs>();
      std::string layout = "NCHW";
      return MaxPool2D(ifm, a->pool_size, a->strides, a->padding, layout, a->ceil_mode);
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }
  
  Expr VisitExpr_(const FunctionNode* fn) final {
    Expr z = VisitExpr_(fn->body.as<CallNode>());
    return FunctionNode::make(FreeVars(z), z, Type(), {});
  }
  
  Expr VisitExpr_(const ConstantNode* op) final {
    std::vector<int64_t> shape;
    int64_t size = 1;
    for (auto x : op->data.Shape()) {
      shape.push_back(x);
      size *= x;
    }
    auto max_iter = std::max_element(shape.begin(), shape.end()); 
    bool is_bias = *max_iter == size;
    if (op->is_scalar()) {
      return ExprMutator::VisitExpr_(op);  
    } else if (is_bias) {
      return ExpandBiasToMatchAxis(ExprMutator::VisitExpr_(op), 4, {1});
    } else if (shape.size() == 4) {
      // HWIO -> OIHW
      float* dl = static_cast<float*>(op->data.ToDLPack()->dl_tensor.data);
      std::vector<float> ret;
      int64_t H = shape[0];
      int64_t W = shape[1]; 
      int64_t I = shape[2];
      int64_t O = shape[3];
      for (int64_t o = 0; o < O; o++) {
        for (int64_t i = 0; i < I; i++) {
          for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < W; w++) {          
              ret.push_back(dl[h*W*I*O+w*I*O+i*O+o]);
            }
          }
        }
      }
      tvm::runtime::NDArray A =
            tvm::runtime::NDArray::Empty({O,I,H,W}, {kDLFloat, 32, 1}, {kDLCPU, 0});
      float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
      std::memcpy(data, &ret[0], sizeof(float)*size);
      return ConstantNode::make(A);  
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }
};

Expr TransformToNCHWLayout(const Expr& expr, const IRModule& mod) {
  // check if input is NHWC
  struct LayoutVisitor : ExprVisitor {
    std::string layout = "NHWC";
    void VisitExpr_(const CallNode* call) final {
      if (call->op == Op::Get("nn.conv2d")) {
        auto a = call->attrs.as<Conv2DAttrs>();
        layout = a->data_layout;
      } else {
        return ExprVisitor::VisitExpr_(call);
      }
    }
  } visitor;
  visitor(expr);
  // TODO : implicitize padding ? or collect nn.pad in gen_hw_ir.cc
  if (visitor.layout == "NHWC") {
    return AlterToNCHWLayout().Mutate(expr);
  } else {
    return expr; 
  }
}

namespace transform {

Pass ToNCHWLayout() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(TransformToNCHWLayout(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "ToNCHWLayout",
                    {ir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.ToNCHWLayout")
.set_body_typed(ToNCHWLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
