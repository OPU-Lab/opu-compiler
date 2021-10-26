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
#include "./gen_hw_ir.h"
#include "./quantize.h"

namespace tvm {
namespace relay {

/*
 * top function to traverse functions and collect info for each operator inside
 */
void IRCollector::Prepare(const Expr& body) {
  // collect info for each node
  this->VisitExpr(body);
  // identify function dependencies
  for (auto item : fn_arg_map_) {
    Func* func = fn_func_map_[item.first];
    for (auto fn_arg : item.second) {
      Func* func_arg = fn_func_map_[fn_arg];
      func->input_funcs.push_back(func_arg);
      func_arg->output_funcs.push_back(func);
    }
  }
  // remove the top function wrapper
  funcs.erase(funcs.begin());
  // sort functions in topological order
  this->MakeTopologicalIndex();
  // update input/output index in opuinfo clss
  for (auto func : funcs) {
    for (auto arg : func->input_funcs) {
      func->info->input_layer.push_back(arg->info->index);
      arg->info->output_layer.push_back(func->info->index);
    }
  }
  // update residual input
  // identify residual input layer index with function input arg index
  for (auto item : func_res_idx_map_) {
    Func* func = item.first;
    for (auto arg_idx : item.second) {
      Func* res_input_func = func->input_funcs[arg_idx];
      func->info->residual_source.push_back(res_input_func->info->index);
    }
  }  
  // debug
  os << "=======================\n";
  for (auto func : funcs) {
    os << "func[" << func->info->index << "]: inputs[";
    for (auto arg : func->input_funcs) {
      os << arg->info->index << " ";
    }
    os << "]\n";
  }
  LOG(INFO) << os.str();
}

void IRCollector::MakeTopologicalIndex() {
  std::unordered_map<Func*, bool> visited;
  for (auto func : funcs) {
    visited[func] = false;
  }
  std::stack<Func*> Stack;
  for (auto func : funcs) {
    if (!visited[func]) {
      TopologicalSortUtil(func, visited, Stack);
    }
  }
  std::vector<Func*> tps_funcs;
  size_t index = 1;
  while (!Stack.empty()) {
    Func* func = Stack.top();
    func->info->index = index++;
    tps_funcs.push_back(func);
    Stack.pop();
  }
  funcs = std::move(tps_funcs);
}

void IRCollector::TopologicalSortUtil(
    Func* func,
    std::unordered_map<Func*, bool> &visited,
    std::stack<Func*> &Stack) {
  visited[func] = true;
  for (auto output : func->output_funcs) {
    if (!visited[output])
      TopologicalSortUtil(output, visited, Stack);
  }
  Stack.push(func);
}

void IRCollector::Update(const tvm::Object* ref) {
  fmap_[ref] = cfunc;
  if (cfunc != nullptr) {
    os << ">>" <<ref << " -> func " << cfunc->info->index << "\n";
  }
}

bool IRCollector::EqFunc(const tvm::Object* e, Func* func) {
  Func* expr_func = nullptr;
  auto it = fmap_.find(e);
  if (it != fmap_.end()) {
    expr_func = it->second;
    // os << "##" << expr_func->info->index << " "
    //    << func->info->index << "\n";
  }
  return expr_func == func;
}

const CallNode* IRCollector::HasConcat(const Expr& body) {
  struct HasConcatVisitor : ExprVisitor {
    const CallNode* concat = nullptr;
    void VisitExpr_(const CallNode* call) final {
      if (call->op == Op::Get("concatenate")) {
        concat = call;
      } else {
        ExprVisitor::VisitExpr_(call);
      }
    }
  } visitor;
  visitor(body);
  return visitor.concat;  
}

size_t IRCollector::FindFirstNonResidualInput(const FunctionNode* fn) {
  struct HasConcatVisitor : ExprVisitor {
    std::vector<const tvm::Object*> add_args;
    void VisitExpr_(const CallNode* call) final {
      if (call->op == Op::Get("add")) {
        for (auto x : call->args) {
          add_args.push_back(x.get());
        }
      } else {
        ExprVisitor::VisitExpr_(call);
      }
    }
  } visitor;
  visitor(GetRef<Expr>(fn));
  size_t i = 0;
  for (i = 0; i < fn->params.size(); i++) {
    auto it = std::find(
        visitor.add_args.begin(), visitor.add_args.end(), fn->params[i].get());
    if (it == visitor.add_args.end()) {
      break;
    }
  }
  return i;
}

void IRCollector::VisitExpr_(const FunctionNode* fn_node) {
  // check if pad only function, then annotate in IR.json for verification
  /*struct IsPadOnlyVisitor : ExprVisitor {
    bool is_pad_only = true;
    void VisitExpr_(const CallNode* call) final {
      if (call->op != Op::Get("nn.pad")) {
        is_pad_only = false;
      }
      ExprVisitor::VisitExpr_(call);
    }
  } visitor;
  visitor(GetRef<Expr>(fn_node));
  if (funcs.size() > 0 && visitor.is_pad_only) {
    ExprVisitor::VisitExpr_(fn_node);
    return;
  }*/
  Func* func = new Func();
  func->info = new OpuInfo();
  // function index (traversal order)
  func->info->index = funcs.size();
  // update input/output shape of function
  if (funcs.size() == 0) {
    const VarNode* op = static_cast<const VarNode*>(fn_node->params[0].get());
    const auto* rtype = op->checked_type().as<TensorTypeNode>();
    top_input_shape = rtype->shape;
  } else {
    // input shape
    if (fn_node->params.size() == 0) {
      for (auto x : top_input_shape) {
        func->info->input_size.push_back(OpuInfo::Value(x));
      }
    } else {
      // for function that has extra residual input, input size should be equal to the 
      // non-residual input instead of the residual input, which could be the first
      // input argument or not
      size_t index = 0;
      if (fn_node->params.size() > 1) {
        index = FindFirstNonResidualInput(fn_node);
      }
      const VarNode* op = static_cast<const VarNode*>(fn_node->params[index].get());
      const auto* rtype = op->checked_type().as<TensorTypeNode>();
      for (auto x : rtype->shape) {
        func->info->input_size.push_back(OpuInfo::Value(x));
      }
      const CallNode* concat = HasConcat(GetRef<Expr>(fn_node));
      // TODO: determine the axis for concatenation
      if (concat != nullptr) {
        auto attrs = concat->attrs.as<ConcatenateAttrs>();
        int axis = attrs->axis;
        for (size_t i = 1; i < fn_node->params.size(); i++){
          const VarNode* op = static_cast<const VarNode*>(fn_node->params[i].get());
          const auto* rtype = op->checked_type().as<TensorTypeNode>();
          func->info->input_size[axis] += OpuInfo::Value(rtype->shape[axis]);
        }
      }
    }
    // output shape
    const auto* ftype = fn_node->body->checked_type().as<TensorTypeNode>();
    for (auto x : ftype->shape) {
      func->info->output_size.push_back(OpuInfo::Value(x));
    }
  }
  // bookkeeping
  func->ref = fn_node;
  fn_func_map_[func->ref] = func;
  funcs.push_back(func);
  os << "FUNC" << func->info->index << ": " << func->ref <<"\n";
  cfunc = func;
  ExprVisitor::VisitExpr_(fn_node);
  cfunc = nullptr;
  os << "END " << func->info->index << "\n";
}

void IRCollector::VisitExpr_(const CallNode* call) {
  if (call->op.as<OpNode>()) {
    os << call->op << "\n";
    this->Update(call);
    if (cfunc != nullptr) {
      OpuInfo* info = cfunc->info;
      const OpNode* op = static_cast<const OpNode*>(call->op.get());
      std::string op_name = op->name;
      // collect operator characteristics
      if (call->op == Op::Get("add")) {
        // check residual 
        // in sub-function, if one operand is var (function input)
        // then it is a residual add, since one input is from other
        // function (layer) output
        bool has_var = false;
        for (auto it = call->args.begin(); it != call->args.end(); ++it) {
          auto arg = *it;
          if (arg.get()->IsInstance<VarNode>()) {
            has_var = true;
            // bookkeep which function input is for residual addition
            func_res_idx_map_[cfunc].push_back(it-call->args.begin());
          } else if (auto pred = arg.as<CallNode>()) {
            if (pred->op == Op::Get("nn.pad")) {
              // must be channel pad
              has_var = true;
              func_res_idx_map_[cfunc].push_back(func_res_idx_map_[cfunc].size());
            }
          }
        }
        if (has_var) {
          os << "[INFO] is residual add\n";
          op_name = "residual_add";
        }
      } else if (call->op == Op::Get("nn.dense")) {
        info->type = 0;
        info->ker_stride = {1, 1};
        info->ker_size = {1, 1};
      } else if (call->op == Op::Get("nn.pad")) {
        info->explicit_pad = 1;
        // assume NCHW
        // collect to explicit_padding_size, overriding 
        // conv2d's implicit padding 
        auto a = call->attrs.as<PadAttrs>();
        for (size_t j = 2; j < a->pad_width.size(); j++) {
          info->explicit_padding_size.push_back(
            OpuInfo::Value(a->pad_width[j][0]));
          info->explicit_padding_size.push_back(
            OpuInfo::Value(a->pad_width[j][1]));
        }
      } else if (call->op == Op::Get("nn.conv2d")) {
        info->type = 1;  // vary with conv type
        auto a = call->attrs.as<Conv2DAttrs>();
        for (auto p : a->padding) {
          info->padding_size.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->strides) {
          info->ker_stride.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->dilation) {
          info->dilation.push_back(OpuInfo::Value(p));
        }
        info->data_layout = a->data_layout;
        info->kernel_layout = a->kernel_layout;
        info->group = a->groups;
        // weight shape
        const auto* wtype = call->args[1]->checked_type().as<TensorTypeNode>();
        for (auto p : wtype->shape) {
          info->ker_size.push_back(OpuInfo::Value(p));
        }
      } else if (call->op == Op::Get("nn.max_pool2d")) {
        info->type = 3;
        info->pooling_type = 1;
        auto a = call->attrs.as<MaxPool2DAttrs>();
        for (auto p : a->padding) {
          info->pool_padding_size.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->strides) {
          info->pooling_stride.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->pool_size) {
          info->pooling_size.push_back(OpuInfo::Value(p));
        }
      } else if (call->op == Op::Get("nn.avg_pool2d")) {
        info->type = 3;
        info->pooling_type = 2;
        auto a = call->attrs.as<AvgPool2DAttrs>();
        for (auto p : a->padding) {
          info->pool_padding_size.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->strides) {
          info->pooling_stride.push_back(OpuInfo::Value(p));
        }
        for (auto p : a->pool_size) {
          info->pooling_size.push_back(OpuInfo::Value(p));
        }
      } else if (call->op == Op::Get("nn.global_avg_pool2d")) {
        info->type = 3;
        info->pooling_type = 2;
        auto a = call->attrs.as<GlobalPool2DAttrs>();
        if (a->layout == "NCHW") {
          opu_int H = static_cast<opu_int>(info->input_size[2]);
          opu_int W = static_cast<opu_int>(info->input_size[3]);
          info->pooling_size.push_back(H);
          info->pooling_stride.push_back(H);
          info->pooling_size.push_back(W);
          info->pooling_stride.push_back(W);
        } else {
          opu_int H = static_cast<opu_int>(info->input_size[1]);
          opu_int W = static_cast<opu_int>(info->input_size[2]);
          info->pooling_size.push_back(H);
          info->pooling_stride.push_back(H);
          info->pooling_size.push_back(W);
          info->pooling_stride.push_back(W);
        }
      } else if (call->op == Op::Get("mean")) {
        if (info->input_size.size() > info->output_size.size()) {
          info->type = 3;
          info->pooling_type = 2;
          opu_int H = static_cast<opu_int>(info->input_size[2]);
          opu_int W = static_cast<opu_int>(info->input_size[2]);
          info->pooling_size.push_back(H);
          info->pooling_stride.push_back(H);
          info->pooling_size.push_back(W);
          info->pooling_stride.push_back(W);
        }
      } else if (call->op == Op::Get("clip")) {
        info->activation_type = 1;
      } else if (call->op == Op::Get("nn.relu")) {
        info->activation_type = 1;
      } else if (call->op == Op::Get("nn.leaky_relu")) {
        info->activation_type = 2;
      } else if (call->op == Op::Get("nn.upsampling")) {
        auto a = call->attrs.as<UpSamplingAttrs>();
        info->upsample = static_cast<opu_int>(a->scale_h);
        info->upsample_method = a->method;
      } else if (call->op == Op::Get("reshape")) {
        if (auto pred = call->args[0].as<CallNode>()) {
          if (pred->op == Op::Get("transpose")) {
            info->channel_shuffle = 3;
          }
        }
      }
      // bookkeep operator names in dfs order 
      info->BookKeepOpOrder(op_name);
    }
  } else {  // func otherwise
    const tvm::Object* func_ref = call->op.get();
    for (auto arg : call->args) {
      const CallNode* sub_call = reinterpret_cast<const CallNode*>(arg.get());
      fn_arg_map_[func_ref].push_back(sub_call->op.get());
    }
  }
  ExprVisitor::VisitExpr_(call);
}

void IRCollector::VisitExpr_(const TupleNode* op) {
  os << "tuple" << "\n";
  this->Update(op);
  ExprVisitor::VisitExpr_(op);
}

void IRCollector::VisitExpr_(const ConstantNode* op) {
  os << "const" << "\n";
  this->Update(op);
  ExprVisitor::VisitExpr_(op);
}

void IRCollector::WriteIR() {
  OpuInfoCollection* vec = new OpuInfoCollection();
  for (auto func : funcs) {
    vec->Add(func->info);
  }
  vec->GlobalCanonicalize();
  if (use_post_padding) {
    vec->ToPostPadding();
    vec->SkipPoolPadding();
  }
  vec->dump2file();
}

/*
 * sanity check
 * make sure layer params follow opu ir definition
 */
void IRCollector::LocalCanonicalize() {
  for (auto func : funcs) {
    func->info->Canonicalize();
  }
}

/*
 * pass top  function
 */
Expr GenIR(const Expr& expr, const IRModule& module, bool quantize, bool use_post_padding) {
  IRCollector irc;
  // collect info
  irc.Prepare(expr);
  // sanity check
  irc.LocalCanonicalize();
  // quantize
  if (quantize) {
    QNN qnn = QNN();
    qnn.fmap_ = irc.fmap_;
    qnn.funcs = irc.funcs;
    qnn.dump_ = true;
    // qnn.dump_unquantized_constant_ = true;
    qnn.Prepare(expr);   
  }
  // generate OPU IR
  irc.use_post_padding = use_post_padding;
  irc.WriteIR();
  return expr;
}

namespace transform {

Pass GenIR(bool quantize, bool use_post_padding) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
    pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(GenIR(f, m, quantize, use_post_padding));
  };
  return CreateFunctionPass(pass_func, 1, "GenIR",
                            {ir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.GenIR")
.set_body_typed(GenIR);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
