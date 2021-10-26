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

#include "./canonicalize_concat_pos.h"

namespace tvm {
namespace relay {

/*
 * top function to collect concat node and its predecessors
 */
void NodeCollector::Prepare(const Expr& body) {
  this->VisitExpr(body);
  LOG(INFO) << os.str();
}

void NodeCollector::VisitExpr_(const TupleNode* op) {
  os << "<TUPLE>\n";
  ExprVisitor::VisitExpr_(op);
}

bool IsSingleInput(const CallNode* call) {
  return call->args.size() == 1;
}
 
void NodeCollector::VisitExpr_(const CallNode* call) {
  ExprVisitor::VisitExpr_(call);
  const Op& concat = Op::Get("concatenate");
  if (call->op == concat) {
    os << call->op << "\n";
    Expr arg = call->args[0];
    // check its tuple succ_concat_mapecessor, whose args are for concat
    if (arg.as<TupleNode>()) {
      os << "Collect concat inputs" << "\n";
      const TupleNode* tuple = arg.as<TupleNode>();
      for (auto u : tuple->fields) {
        if (u.as<CallNode>()) {
          // TODO(td):
          // tuple inputs can only be CallNode instead of ConcatNode for now
          const CallNode* node = u.as<CallNode>();
          // bookkeep concat tuple input exprs
          concat_inputs_map[call].push_back(node);
          expr_map[node] = u;
        }
      }
    }
  } else {
    // check if one node is direct successor of concat
    for (auto arg : call->args) {
      if (const CallNode* succ_concat_map_call = arg.as<CallNode>()) {
        if (succ_concat_map_call->op == concat) {
          // auto it = memo_.find(succ_concat_map_call);
          // annotate node as movable only if it is the 
          // only successor of it proceeding concat
          // otherwise it can be a dangerous transform
          if (//it == memo_.end() &&
              IsChannelWise(call) && 
              IsSingleInput(call)) {
            succ_concat_map[call] = succ_concat_map_call;
            os << "\n after concat:" << call->op << "\n";
          }
        }
      }
    }
  }
  memo_[call] = true;
}

/* 
 * determine whether an operator is channel-wise
 * if so, then is can be moved before concat
 */
bool NodeCollector::IsChannelWise(const CallNode* call) {
  // pooling applies
  if (call->op == Op::Get("nn.max_pool2d") ||
      call->op == Op::Get("nn.avg_pool2d") ||
      call->op == Op::Get("nn.global_avg_pool2d")) {
    // OpPatternKind=4 for pooling and conv2d as "kOutEWiseFusable"
    return true;
  }
  // injective (element-wise) operator applies
  static auto fpattern = Op::GetAttr<TOpPattern>("TOpPattern");
  if (const OpNode* opnode = call->op.as<OpNode>()) {
    OpPatternKind op_pattern =
      static_cast<OpPatternKind>(fpattern[GetRef<Op>(opnode)]);
    if (op_pattern <= kInjective) {
      return true;
    }
  }
  return false;
}

/*
 * top function to mutate expr
 */
Expr GraphMutator::Transform(const Expr& body) {
  os << "\n\n[CanonicalizeConcatPos Pass]\n";
  Expr res = this->Mutate(body);
  os << "[Done]\n";
  LOG(INFO) << os.str();
  return res;
}

Expr GraphMutator::VisitExpr_(const CallNode* call) {
  Expr new_expr = ExprMutator::VisitExpr_(call);
  // mutate at direct concat successor
  if (succ_concat_map.find(call) != succ_concat_map.end()) {
    os << "Move " << call->op
       << " before " << succ_concat_map[call]->op << "\n";
    // move call before concat
    Array<Expr> inp;
    // get concat input exprs
    for (auto u : concat_inputs_map[succ_concat_map[call]]) {
      os << call->op <<"\n";
      Expr e = CallNode::make(call->op,
        Array<Expr>{ExprMutator::VisitExpr_(u)}, call->attrs, call->type_args);
      inp.push_back(e);
    }
    const auto* param = succ_concat_map[call]->attrs.as<ConcatenateAttrs>();
    return MakeConcatenate(TupleNode::make(inp), param->axis);
  } else {
    return new_expr;
  }
}

/*
 * pass top function
 */
Expr CanonicalizeConcatPos(const Expr& expr, const IRModule& module) {
  NodeCollector collector = NodeCollector();
  GraphMutator gm = GraphMutator();
  Expr ret = expr;
  // after concat: activation, pad, pool
  for (int i = 0; i < 3; i++) {
    collector.Prepare(ret);
    if (!collector.succ_concat_map.empty()) {
      // there is node between concat and conv/fc
      gm.concat_inputs_map = collector.concat_inputs_map;
      gm.succ_concat_map = collector.succ_concat_map;
      ret = gm.Transform(ret);
      collector.concat_inputs_map.clear();
      collector.succ_concat_map.clear();
    } else {
      break;
    }
  }
  return ret;
}

namespace transform {

Pass CanonicalizeConcatPos() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
    pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(CanonicalizeConcatPos(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeConcatPos",
                            {ir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.CanonicalizeConcatPos")
.set_body_typed(CanonicalizeConcatPos);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
