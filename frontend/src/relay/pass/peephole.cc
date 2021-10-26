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

class PatternMatcher1 : public ExprVisitor {
 public:
  std::unordered_map<const CallNode*, const CallNode*> add_mul_map;
  std::unordered_map<const CallNode*, const CallNode*> mul_add_map;
  void VisitExpr_(const CallNode* call) final {
    for (auto arg : call->args) {
      if (arg.get()->IsInstance<CallNode>()) {
        auto pred = reinterpret_cast<const CallNode*>(arg.get());
        bool is_pred_conv = pred->op == Op::Get("nn.conv2d") ||
                            pred->op == Op::Get("nn.conv2d_transpose");
        if (is_pred_conv && call->op == Op::Get("add")) {
          add_mul_map[call] = nullptr;  
        } else if (pred->op == Op::Get("add") && call->op == Op::Get("multiply")) {
          mul_add_map[call] = pred;  
        }
      }
    }
    ExprVisitor::VisitExpr_(call);
  }
  // get conv2d - [add - mul] - add - 
  void Prepare(const Expr& body) {
    this->VisitExpr(body);
    std::unordered_map<const CallNode*, bool> keep;
    for (auto item : mul_add_map) {
      auto it = add_mul_map.find(item.second);
      if (it != add_mul_map.end()) {
        keep[item.first] = true;
      } else {
        keep[item.first] = false;
      }
    }
    for (auto item : keep) {
      if (!item.second) {
        mul_add_map.erase(item.first);
      }
    }
  }
};
class PatternTransformer1 : public ExprMutator {
 public:
  std::unordered_map<const CallNode*, const CallNode*> mul_add_map;
  Expr VisitExpr_(const CallNode* call) final {
    Expr new_expr = ExprMutator::VisitExpr_(call);
    auto it = mul_add_map.find(call);
    if (it != mul_add_map.end()) {
      auto add = it->second;
      Expr inp_add = VisitExpr_(
        reinterpret_cast<const CallNode*>(add->args[0].get()));
      auto add_param = GetConst(add);
      auto mul_param = GetConst(call);
      Expr m = ConstMul(add_param, mul_param);
      //return Add(Multiply(inp_add, call->args[1]), ExpandBiasToMatchAxis(m, 4, {1}));
      return Add(inp_add, ExpandBiasToMatchAxis(m, 4, {1}));
    }
    return new_expr;
  }
  const ConstantNode* GetConst(const CallNode* call) {
    const CallNode* tmp = call;
    while (!tmp->args[tmp->args.size() - 1].get()->IsInstance<ConstantNode>()) {
      assert(call->args[0].get()->IsInstance<CallNode>());
      tmp = reinterpret_cast<const CallNode*>(call->args[tmp->args.size() - 1].get());  
    }
    return reinterpret_cast<const ConstantNode*>(tmp->args[tmp->args.size() - 1].get());
  }
  Expr ConstMul(const ConstantNode* x, const ConstantNode* y) {
    // Not Implemented!
    return GetRef<Expr>(x);  
  }
};

class PatternTransformer2 : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call) final {
    Expr new_expr = ExprMutator::VisitExpr_(call);
    if (isBiasAdd(call)) {
      if (call->args[0].get()->IsInstance<CallNode>()) {
        auto pred = reinterpret_cast<const CallNode*>(call->args[0].get());
        if (isBiasAdd(pred)) {
          std::cout << "Found biasadd in row!\n";
          // Not Implemented!
          Expr param = call->args[1];//Add(call->args[1], pred->args[1]);
          Expr inp = VisitExpr_(
            reinterpret_cast<const CallNode*>(pred->args[0].get()));
          return Add(inp, param);
        }
      }
    }
    return new_expr;
  }
  bool isBiasAdd(const CallNode* call) {
    bool is_add = (call->op == Op::Get("add"));
    if (!is_add) {
      return false;
    }
    bool has_const_operand = call->args[1].get()->IsInstance<ConstantNode>();
    if (!has_const_operand && call->args[1].get()->IsInstance<CallNode>()) {
      auto arg1 = reinterpret_cast<const CallNode*>(call->args[1].get());
      if (arg1->args.size() == 1 && arg1->args[0].get()->IsInstance<ConstantNode>()) {
        has_const_operand = true;
      }
    }
    return is_add && has_const_operand;
  }
};

/*
 * residual input = concat(layer 1 output, layer 2 output)
 *
 * %3 = concatenate(%1, %2)
 * %4 = nn.conv2d(%0, constant)
 * %5 = add(%3, %4) // residual add
 *
 * =>
 *
 * %4 = nn.conv2d(%0, constant)
 * %5 = split(%4, C1)
 * %6 = add(%5, %1)
 * %7 = split(%4, C2)
 * %8 = add(%7, %2)
 * %9 = concatenate(%7, %8)
 */
class PatternTransformer3 : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call) final {
    Expr new_expr = ExprMutator::VisitExpr_(call);
    if (IsEleAdd(call)) {
      if (auto cc = call->args[0].as<CallNode>()) {
        if (cc->op == Op::Get("concatenate")) {
          const auto* param = cc->attrs.as<ConcatenateAttrs>();
          CHECK_EQ(param->axis, 1);
          auto tuple = cc->args[0].as<TupleNode>();
          CHECK_EQ(tuple->fields.size(), 2);
          auto t0 = tuple->fields[0].as<CallNode>();
          const auto* rtype = t0->checked_type().as<TensorTypeNode>();
          int c0 = ToInt(rtype->shape[param->axis]);
          auto t1 = tuple->fields[1].as<CallNode>();
          rtype = t1->checked_type().as<TensorTypeNode>();
          int c1 = ToInt(rtype->shape[param->axis]);
          // Transform
          auto inp = this->Mutate(call->args[1]);
          auto t0_cpad = MakePad(this->Mutate(tuple->fields[0]), {0, c1}, param->axis);
          auto ret = Add(t0_cpad, inp);
          auto t1_cpad = MakePad(this->Mutate(tuple->fields[1]), {c0, 0}, param->axis);
          ret = Add(t1_cpad, ret);
          return ret;
        }
      }
    }
    return new_expr;
  }
  Expr MakePad(Expr e, std::vector<int> pw, int axis) {
    Array<Array<IndexExpr>> pad_width;
    CHECK_LT(axis, 4);
    for (int i = 0; i < 4; i++) {
      Array<IndexExpr> tmp;
      for (int j = 0; j < 2; j++) {
        if (i == axis) {
          tmp.push_back(IndexExpr(pw[j]));
        } else {
          tmp.push_back(IndexExpr(0));
        }
      }
      pad_width.push_back(tmp);
    }
    double pad_value = 0.0;
    std::string pad_mode = "constant";
    return Pad(e, pad_width, pad_value, pad_mode);
  }
  int ToInt(const PrimExpr& e) {
    return static_cast<int>(static_cast<const IntImmNode*>(e.get())->value);
  }
  bool IsConst(const Expr& e) {
    if (e.get()->IsInstance<ConstantNode>()) {
      return true;
    } else if (e.get()->IsInstance<CallNode>()) {
      auto call = reinterpret_cast<const CallNode*>(e.get());
      if (call->args.size() == 1 &&
          call->args[0].get()->IsInstance<ConstantNode>()) {
        return true;
      }
    }
    return false;
  }
  bool IsEleAdd(const CallNode* call) {
    if (call->op != Op::Get("add")) {
      return false;
    }
    if (!IsConst(call->args[0]) && !IsConst(call->args[1])) {
      return true;
    } else {
      return false;
    }
  }
};

/*
 * Apply transpose operator to constant weight instead of mainline
 * 
 * %0 ty=Tensor[(1,128,1,347)]
 * %1 = transpose(%0, axes=[0,2,3,1]) ty=Tensor[(1,1,347,128)]
 * %2 = nn.batch_flatten(%1) ty=Tensor[(1,44416)]
 * mm_weight ty=Tensor[(66, 44416)]
 * %3 = nn.dense(%2, mm_weight) ty=Tensor[(1,66)]
 *
 * ->
 *
 * %0 ty=Tensor[(1,128,1,347)]
 * %1 = nn.batch_flatten(%0) ty=Tensor[(1,44416)]
 * %2 = reshape(mm_weight, new_shape=[66, 128, 347])
 * %3 = transpose(%2, axes=[0,2,1]) ty=Tensor[(66, 44416)]
 * %4 = nn.dense(%2, %3) ty=Tensor[(1,66)]
 */
class PatternTransformer4 : public ExprMutator {
 public:
  std::unordered_map<const ConstantNode*, std::vector<int64_t>> tmap_;
  Expr VisitExpr_(const CallNode* call) final {
    if (call->op == Op::Get("nn.dense")) {
      if (auto arg0 = call->args[0].as<CallNode>()) {
        if (auto pred = IsTransposed(arg0)) {
          auto arg1 = call->args[1].as<ConstantNode>();
          CHECK(arg1 != nullptr);
          const auto* rtype = pred->checked_type().as<TensorTypeNode>();
          for (auto x : rtype->shape) {
            int64_t value = ToInt(x);
            if (value != 1) {
              tmap_[arg1].push_back(value);
            }
          }
          Expr arg0_new = ExprMutator::VisitExpr_(pred);
          arg0_new = CallNode::make(Op::Get("nn.batch_flatten"), {arg0_new}, Attrs(), {});
          Expr arg1_new = VisitExpr_(arg1);
          return CallNode::make(call->op,
          {arg0_new, arg1_new}, call->attrs, call->type_args);
        }
      }
    }
    return ExprMutator::VisitExpr_(call);
  }
  int64_t ToInt(const PrimExpr& e) {
    return static_cast<int64_t>(static_cast<const IntImmNode*>(e.get())->value);
  }
  Expr VisitExpr_(const ConstantNode* op) {
    auto it = tmap_.find(op);
    if (it != tmap_.end()) {
      std::cout << "[Peephole] Found MM weight that need to be transposed\n";
      CHECK_EQ(it->second.size(), 2) << "Only 2d before flatten is supported for now\n";
      float* dl = static_cast<float*>(op->data.ToDLPack()->dl_tensor.data);
      std::vector<float> ret;
      int64_t O = op->data.Shape()[0];
      int64_t I = op->data.Shape()[1];
      int64_t I0 = it->second[0]; 
      int64_t I1 = it->second[1]; 
      CHECK_EQ(I, I0 * I1);
      for (int64_t o = 0; o < O; o++) {
        for (int64_t i = 0; i < I0; i++) {
          for (int64_t j = 0; j < I1; j++) {
            ret.push_back(dl[o * I + j * I0 + i]);
          }
        }
      }
      tvm::runtime::NDArray A =
            tvm::runtime::NDArray::Empty({O, I}, {kDLFloat, 32, 1}, {kDLCPU, 0});
      float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
      std::memcpy(data, &ret[0], sizeof(float) * O * I);
      return ConstantNode::make(A);
    } else {
      return GetRef<Expr>(op);
    }
  }
  const CallNode* IsTransposed(const CallNode* call) {
    if (call->op == Op::Get("nn.batch_flatten")) {
      if (auto pred = call->args[0].as<CallNode>()) {
        if (pred->op == Op::Get("transpose")) {
          return pred->args[0].as<CallNode>();
        }
      }    
    } else if (call->op == Op::Get("transpose")) {
      return call;  
    }
    return nullptr;
  }
};

/*
 * Convert fc (with small input channels e.g.,1024) to 1x1 conv
 */
class PatternTransformer5 : public ExprMutator {
 public:
  int input_channel_thres {2048};
  Expr VisitExpr_(const CallNode* call) final {
    if (call->op == Op::Get("nn.dense")) {
      auto arg0 = call->args[0].as<CallNode>();
      if (arg0->op == Op::Get("nn.batch_flatten")) {
        auto arg00 = arg0->args[0].as<CallNode>();
        if (arg00->op == Op::Get("transpose")) {
          auto data = arg00->args[0];
          // weight (Co, Ci) -> (Co, Ci, 1, 1)
          auto wconst = call->args[1].as<ConstantNode>();
          float* dl = static_cast<float*>(wconst->data.ToDLPack()->dl_tensor.data);
          std::vector<float> ret;
          int64_t O = wconst->data.Shape()[0];
          int64_t I = wconst->data.Shape()[1];
          for (int64_t o = 0; o < O; o++) {
            for (int64_t i = 0; i < I; i++) {
              ret.push_back(dl[o * I + i]);
            }
          }  
          tvm::runtime::NDArray A =
            tvm::runtime::NDArray::Empty({O, I, 1, 1}, {kDLFloat, 32, 1}, {kDLCPU, 0});
          float* wdata = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
          std::memcpy(wdata, &ret[0], sizeof(float) * O * I);
          auto weight = ConstantNode::make(A);
          // assertion
          int Ci = static_cast<int>(I);
          int Co = static_cast<int>(O);
          if (Ci > input_channel_thres) {
            return ExprMutator::VisitExpr_(call);  
          }
          std::cout << "[Peephole] Found FC Ci <= " << input_channel_thres
                    << " to be transformed to pointwise conv2d\n";
          // (1, Ci, 1, 1) -> 1x1 conv -> (1, Co, 1, 1)
          Array<IndexExpr> strides = {relay::IndexExpr(1), relay::IndexExpr(1)};
          auto padding = {relay::IndexExpr(0), relay::IndexExpr(0),
                          relay::IndexExpr(0), relay::IndexExpr(0)};
          auto dilation = {relay::IndexExpr(1), relay::IndexExpr(1)};
          int groups = 1;
          auto channels = relay::IndexExpr(Co);
          auto kernel_size = {relay::IndexExpr(1), relay::IndexExpr(1)};
          std::string target_layout = "NCHW";
          std::string data_layout = target_layout;
          std::string kernel_layout = (target_layout == "NCHW")? "OIHW" : "HWIO";
          std::string out_layout = data_layout;
          auto out_dtype = runtime::DataType::Float(32);
          data = Conv2D(data, weight, strides, padding, dilation, groups,
                              channels, kernel_size, data_layout, kernel_layout,
                              out_layout, out_dtype);
          // reshape -> (1, Co)
          data = Reshape(data, {1, Co});
          return data;
        }
      }
    }
    return ExprMutator::VisitExpr_(call);
  }
  int64_t ToInt(const PrimExpr& e) {
    return static_cast<int64_t>(static_cast<const IntImmNode*>(e.get())->value);
  }
};

/*
 * Skip output activation (e.g., softmax)
 * The purpose is to apply correct quantization analysis for the final layer
 * (since hw does not have softmax at the end of each layer)
 */
class OutputActivationSkipper : public ExprMutator {
 public:
  bool check {true};
  Expr VisitExpr_(const CallNode* call) final {
    if (check && call->op == Op::Get("nn.softmax")) {
      auto arg0 = call->args[0].as<CallNode>();
      check = false;
      std::cout << "[Peephole] Remove output nn.softmax\n";
      return ExprMutator::VisitExpr_(arg0);
    }
    check = false;
    return ExprMutator::VisitExpr_(call);
  }
};

class StaticProfiler : public ExprVisitor {
 public:
  int op_cnt = 0;
  void VisitExpr_(const CallNode* call) final {
    if (call->op != Op::Get("expand_dims")) {
      op_cnt++;  
    }
    ExprVisitor::VisitExpr_(call);
  }
};

Expr PeepholeOpt(const Expr& expr, const IRModule& mod) {
  StaticProfiler sp;
  sp.VisitExpr(expr);
  std::cout << "Statistics: op count: " << sp.op_cnt << "\n";  
  
  OutputActivationSkipper pm0;
  Expr e = pm0.Mutate(expr);
  
  PatternMatcher1 pm1;
  pm1.Prepare(e);
  PatternTransformer1 pt1;
  pt1.mul_add_map = pm1.mul_add_map;
  e = pt1.Mutate(e);
  
  PatternTransformer2 pt2;
  e = pt2.Mutate(e);

  PatternTransformer3 pt3;
  e = pt3.Mutate(e);
  
  PatternTransformer5 pt5;
  e = pt5.Mutate(e);
  return e;
}

namespace transform {

Pass PeepholeScan() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(PeepholeOpt(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "PeepholeScan",
                    {ir::StringImmNode::make("InferType")});
}

Pass Peephole() {
  Pass pass = Sequential(
      {PeepholeScan(), BackwardFoldScaleAxis(), ForwardFoldScaleAxis(), FoldConstant()},
      "Peephole");
  return pass;
}

TVM_REGISTER_GLOBAL("relay._transform.Peephole")
.set_body_typed(Peephole);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
