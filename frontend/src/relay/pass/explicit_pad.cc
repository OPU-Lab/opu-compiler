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


#include "./explicit_pad.h"

namespace tvm {
namespace relay {

/*
 * top function to collect operators with implicit padding 
 */
void ImplicitPadCollector::Prepare(const Expr& body) {
  os << "\n\n[ExplciitPad Pass - Indentify Implicit Pad]\n";
  this->VisitExpr(body);
  LOG(INFO) << os.str();
}

void ImplicitPadCollector::VisitExpr_(const CallNode* call) {
  // check padding for conv/pool
  const Op& conv2d = Op::Get("nn.conv2d");
  const Op& maxpool2d = Op::Get("nn.max_pool2d");
  const Op& avgpool2d = Op::Get("nn.avg_pool2d");
  if (call->op == conv2d) {
    auto a = call->attrs.as<Conv2DAttrs>();
    if (CheckImplicitPad(a->padding)) {
      os << call->op << "\n";
      os << "  pad:" << a->padding << "\n";
      pad_node_size_map[call] = PadTransform(a->padding, a->data_layout);
    }
  } else if (call->op == maxpool2d) {
    auto a = call->attrs.as<MaxPool2DAttrs>();
    if (CheckImplicitPad(a->padding)) {
      os << call->op << "\n";
      os << "  pad:" << a->padding << "\n";
      pad_node_size_map[call] = PadTransform(a->padding, a->layout);
    }
  } else if (call->op == avgpool2d) {
    auto a = call->attrs.as<AvgPool2DAttrs>();
    if (CheckImplicitPad(a->padding)) {
      os << call->op << "\n";
      os << "  pad:" << a->padding << "\n";
      pad_node_size_map[call] = PadTransform(a->padding, a->layout);
    }
  }
  ExprVisitor::VisitExpr_(call);
}

/*
 * check if non-zero pad width exists
 */
bool ImplicitPadCollector::CheckImplicitPad(Array<PrimExpr> padding) {
  bool implicit = false;
  for (auto e : padding) {
    if (OpuInfo::Value(e) != 0) {
      implicit = true;
      break;
    }
  }
  return implicit;
}

/*
 * conv_attrs->padding:[1,1,1,1] -> pad_attrs->pad_width:[[0,0],[0,0],[1,1],[1,1]]
 */
Array<Array<PrimExpr>> ImplicitPadCollector::PadTransform(
  Array<PrimExpr> padding, std::string layout) {
  Array<Array<PrimExpr>> ret;
  if (layout == "NCHW") {
    ret = {{0, 0}, {0, 0}, {padding[0], padding[2]}, {padding[1], padding[3]}};
  } else if (layout == "NHWC") {
    ret = {{0, 0}, {padding[0], padding[2]}, {padding[1], padding[3]}, {0, 0}};
  } else {
    std::cout << "[ERROR] Unrecognized layout: " << layout << "\n";
    exit(1);
  }
  return ret;
}


// Run the transform
/*
 * top function to mutate expr
 */
Expr GraphMutatorPad::Transform(const Expr& body) {
  os << "\n\n[ExplciitPad Pass - Mutate]\n";
  Expr res = this->Mutate(body);
  os << "[DONE]\n";
  LOG(INFO) << os.str();
  return res;
}

Array<IndexExpr> GraphMutatorPad::MakeZeroPadding(size_t size) {
  Array<IndexExpr> padding_zero;
  for (size_t i = 0; i < size; i++) {
    padding_zero.push_back(PrimExpr(0));
  }
  return padding_zero;
}

Expr GraphMutatorPad::InsertPad(Expr data, Array<Array<PrimExpr>> pad_width) {
  auto attrs = make_object<PadAttrs>();
  attrs->pad_width = std::move(pad_width);
  return CallNode::make(Op::Get("nn.pad"), {data}, Attrs(attrs), {});
}

Expr GraphMutatorPad::VisitExpr_(const CallNode* call) {
  // run ExprMutator::VisitExpr_ to avoid duplicate expr!
  ExprMutator::VisitExpr_(call);
  auto it = pad_node_size_map.find(call);
  if (it != pad_node_size_map.end()) {
    if (it->first->op == Op::Get("nn.conv2d")) {
      // insert pad before conv2d
      Expr ifm;
      if (memo_.find(call->args[0]) != memo_.end()) {
        ifm = memo_[call->args[0]];
      } else {
        ifm = ExprMutator::VisitExpr_(call->args[0].as<CallNode>());
      }
      Expr weight = call->args[1];
      ifm = InsertPad(ifm, it->second);
      // make new zero padding
      auto op_attrs = call->attrs.as<Conv2DAttrs>();
      Array<IndexExpr> padding_zero = MakeZeroPadding(op_attrs->padding.size());
      // make new conv expr
      return Conv2D(ifm, weight, op_attrs->strides,
                    padding_zero, op_attrs->dilation, op_attrs->groups,
                    op_attrs->channels, op_attrs->kernel_size,
                    op_attrs->data_layout, op_attrs->kernel_layout,
                    op_attrs->out_layout, op_attrs->out_dtype);
    } else if (it->first->op == Op::Get("nn.max_pool2d")) {
      // insert pad before max_pool2d
      Expr ifm;
      if (memo_.find(call->args[0]) != memo_.end()) {
        ifm = memo_[call->args[0]];
      } else {
        ifm = ExprMutator::VisitExpr_(call->args[0].as<CallNode>());
      }
      ifm = InsertPad(ifm, it->second);
      // make new zero padding
      auto op_attrs = call->attrs.as<MaxPool2DAttrs>();
      Array<IndexExpr> padding_zero = MakeZeroPadding(op_attrs->padding.size());
      // make new max pool expr
      return MaxPool2D(ifm, op_attrs->pool_size, op_attrs->strides,
                       padding_zero, op_attrs->layout, op_attrs->ceil_mode);
    } else if (it->first->op == Op::Get("nn.avg_pool2d")) {
      // insert pad before avg_pool2d
      Expr ifm;
      if (memo_.find(call->args[0]) != memo_.end()) {
        ifm = memo_[call->args[0]];
      } else {
        ifm = ExprMutator::VisitExpr_(call->args[0].as<CallNode>());
      }
      ifm = InsertPad(ifm, it->second);
      // make new zero padding
      auto op_attrs = call->attrs.as<AvgPool2DAttrs>();
      Array<IndexExpr> padding_zero = MakeZeroPadding(op_attrs->padding.size());
      // make new avg pool expr
      return AvgPool2D(ifm, op_attrs->pool_size, op_attrs->strides,
                       padding_zero, op_attrs->layout, op_attrs->ceil_mode,
                       op_attrs->count_include_pad);
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  } else {
    return ExprMutator::VisitExpr_(call);
  }
}

/*
 * pass top function 
 */
Expr ExplicitPad(const Expr& expr, const IRModule& module) {
  // collect implicit padding
  ImplicitPadCollector collector = ImplicitPadCollector();
  collector.Prepare(expr);
  // make them explicit
  GraphMutatorPad gm = GraphMutatorPad();
  gm.pad_node_size_map = collector.pad_node_size_map;
  Expr ret = gm.Transform(expr);
  return ret;
}

namespace transform {

Pass ExplicitPad() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
    pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(ExplicitPad(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "ExplicitPad",
                            {ir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.ExplicitPad")
.set_body_typed(ExplicitPad);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
