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
#include "./fuse_ops_hw.h"

namespace tvm {
namespace relay {

/* 
 * top function to build DFG(Data Flow Graph) from Relay Expr
 */
DFG DFG::Creator::Prepare(const Expr& body) {
  this->Update(body, nullptr);
  this->VisitExpr(body);
  this->AnnotateInputs();
  LOG(INFO) << os.str();
  return std::move(graph_);
}

/*
 * update DFG::Node inputs from collected outputs (colleted when building the dfg)
 */
void DFG::Creator::AnnotateInputs() {
  for (auto node : graph_.post_dfs_order) {
    for (auto succ : node->succ) {
      succ->pred.push_back(node);
    }
  }
}

/*
 * add tvm node to node map and update its outputs
 */
void DFG::Creator::Update(const Expr& node, DFG::Node* succ) {
  const tvm::Object* key = node.get();
  DFG::Node* current;
  auto it = graph_.node_map.find(key);
  if (it != graph_.node_map.end()) {
    current = it->second;
  } else {
    current = new DFG::Node();
    graph_.node_map[key] = current;
  }
  if (succ != nullptr) {
    current->succ.push_back(succ);
  }
}

/*
 * create DFG::Node for tvm node
 * also check if operator is applied on two fm input (e.g. residual add)
 */
void DFG::Creator::AddNode(const tvm::Object* key) {
  auto it = graph_.node_map.find(key);
  CHECK(it != graph_.node_map.end())
      << "Cannot find node " << GetRef<ObjectRef>(key);
  DFG::Node* node = it->second;
  CHECK(node->ref == nullptr);
  node->ref = key;
  node->index = graph_.post_dfs_order.size();
  graph_.post_dfs_order.push_back(node);
  if (key->IsInstance<CallNode>()) {
    const CallNode* call = static_cast<const CallNode*>(key);
    if (call->op == Op::Get("add") ||
        call->op == Op::Get("multiply") ||
        call->op == Op::Get("divide")) {
      // check element-wise
      const auto* type = call->args[0]->checked_type().as<TensorTypeNode>();
      size_t dim_0 = type->shape.size();
      type = call->args[1]->checked_type().as<TensorTypeNode>();
      size_t dim_1 = type->shape.size();
      if (dim_0 == dim_1) {
        node->isEW = true;
      }
    }
  }
}

// Post order tree
/*
 * utility functions copied from fuse_ops.cc 
 * replace with custom AddNode() above
 */
void DFG::Creator::VisitExpr_(const FunctionNode* op) {
  for (auto param : op->params) {
    this->Update(param, nullptr);
  }
  this->Update(op->body, nullptr);
  ExprVisitor::VisitExpr_(op);
}

void DFG::Creator::VisitExpr_(const ConstantNode* op) {
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const CallNode* call) {
  CHECK(graph_.node_map.count(call));
  Node* node = graph_.node_map.at(call);
  this->Update(call->op, nullptr);
  for (size_t i = 0; i < call->args.size(); ++i) {
    this->Update(call->args[i], node);
  }
  ExprVisitor::VisitExpr_(call);
  this->AddNode(call);
}

void DFG::Creator::VisitExpr_(const TupleNode* op) {
  CHECK(graph_.node_map.count(op));
  Node* tuple_node = graph_.node_map.at(op);
  for (const Expr& field : op->fields) {
    if (field->checked_type().as<TensorTypeNode>()) {
      this->Update(field, tuple_node);
    } else {
      this->Update(field, nullptr);
    }
  }
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const TupleGetItemNode* op) {
  auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
  CHECK(tuple_type);
  bool has_non_tensor = false;
  for (auto ty : tuple_type->fields) {
    if (!ty.as<TensorTypeNode>()) {
      has_non_tensor = true;
      break;
    }
  }
  if (has_non_tensor) {
    this->Update(op->tuple, nullptr);
  } else {
    CHECK(graph_.node_map.count(op));
    Node* node = graph_.node_map.at(op);
    this->Update(op->tuple, node);
  }
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const VarNode* op) {
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const LetNode* op) {
  // do not fuse through let.
  this->Update(op->var, nullptr);
  this->Update(op->value, nullptr);
  this->Update(op->body, nullptr);
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const IfNode* op) {
  // do not fuse through if.
  this->Update(op->cond, nullptr);
  this->Update(op->true_branch, nullptr);
  this->Update(op->false_branch, nullptr);
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const RefCreateNode* op) {
  this->Update(op->value, nullptr);
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const RefReadNode* op) {
  this->Update(op->ref, nullptr);
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const RefWriteNode* op) {
  this->Update(op->ref, nullptr);
  this->Update(op->value, nullptr);
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Creator::VisitExpr_(const MatchNode* op) {
  this->Update(op->data, nullptr);
  for (const Clause& c : op->clauses) {
    this->Update(c->rhs, nullptr);
  }
  ExprVisitor::VisitExpr_(op);
  this->AddNode(op);
}

void DFG::Dump() {
  std::ostringstream os;
  os << "\n[DFG in post dfs order]";
  for (size_t i = 0; i < post_dfs_order.size(); ++i) {
    Node* node = post_dfs_order[i];
    os << "node[" << node->index << "], ";
    if (node->ref->IsInstance<CallNode>()) {
      const CallNode* call = static_cast<const CallNode*>(node->ref);
      os << call->op;
      if (call->op == Op::Get("expand_dims")) {
        const auto* type = call->checked_type().as<TensorTypeNode>();
        os << " [" << type->shape << "]";
      }
    } else if (node->ref->IsInstance<ConstantNode>()) {
      const ConstantNode* op = static_cast<const ConstantNode*>(node->ref);
      os << "constant [";
      for (auto x : op->data.Shape()) {
        os << PrimExpr(static_cast<int>(x)) << " ";
      }
      os << "]";
    } else if (node->ref->IsInstance<VarNode>()) {
      const VarNode* op = static_cast<const VarNode*>(node->ref);
      const auto* rtype = op->checked_type().as<TensorTypeNode>();
      os << "var " << rtype->shape;
    } else if (node->ref->IsInstance<TupleNode>()) {
      const TupleNode* op = static_cast<const TupleNode*>(node->ref);
      os << "tuple ";
      for (const Expr& field : op->fields) {
        auto it = node_map.find(field.get());
        if (it != node_map.end()) {
          os << it->second->index << " ";
        } else {
          os << "\\ ";
        }
      }
    } else {
      os << "### Not Considered ###";
    }
    os << "\tsucc:[";
    for (auto succ : node->succ) {
      os << succ->index << " ";
    }
    os << "]\n";
  }
  LOG(INFO) << os.str();
}

/* 
 * HDFG(Hardware Data Flow Graph) 
 */
HDFG HDFG::Create() {
  HDFG hdfg;
  // list all hardware modules
  auto concat = hdfg.AddNode(MEM_RD);
  auto ipa = hdfg.AddNode(IPA);
  auto add = hdfg.AddNode(ADDER);
  auto add_pre = hdfg.AddNode(ADDER);
  auto mul_pre = hdfg.AddNode(MULTIPLIER);
  auto mul = hdfg.AddNode(MULTIPLIER);
  auto compare = hdfg.AddNode(COMPARATOR);
  auto pool_pad = hdfg.AddNode(PAD);
  auto pool = hdfg.AddNode(POOL);
  auto res_add = hdfg.AddNode(RES_ADDER);
  auto pad = hdfg.AddNode(PAD);
  auto upsample = hdfg.AddNode(UPSAMPLE);
  // describe data flow
  hdfg.root = concat;
  hdfg.LinkFromTo(concat, add_pre);
  hdfg.LinkFromTo(add_pre, mul_pre);
  hdfg.LinkFromTo(mul_pre, pad);
  hdfg.LinkFromTo(pad, ipa);
  hdfg.LinkFromTo(ipa, mul);
  hdfg.LinkFromTo(mul, add);
  hdfg.LinkFromTo(add, compare);
  hdfg.LinkFromTo(add, pool_pad);
  hdfg.LinkFromTo(add, res_add);
  hdfg.LinkFromTo(pool_pad, pool);
  hdfg.LinkFromTo(compare, pool);
  hdfg.LinkFromTo(compare, res_add);
  hdfg.LinkFromTo(pool, compare);
  hdfg.LinkFromTo(pool, res_add);
  hdfg.LinkFromTo(res_add, compare);
  hdfg.LinkFromTo(res_add, pool);
  hdfg.LinkFromTo(compare, upsample);
  hdfg.LinkFromTo(pool, upsample);
  hdfg.LinkFromTo(res_add, upsample);
  return hdfg;
}

HDFG::Node* HDFG::AddNode(HWMODULE module) {
  Node* node = new Node();
  node->module = module;
  node->index = nodes.size();
  nodes.push_back(node);
  return node;
}

void HDFG::LinkFromTo(Node* current, Node* pred) {
  current->succ.push_back(pred);
}

std::string HDFG::Node::ModuleName() {
  std::ostringstream os;
  if (module == MEM_RD) {
    os << "mem_rd ";
  } else if (module == IPA) {
    os << "ipa ";
  } else if (module == ADDER) {
    os << "adder ";
  } else if (module == MULTIPLIER) {
    os << "multiplier ";
  } else if (module == COMPARATOR) {
    os << "comparator ";
  } else if (module == POOL) {
    os << "pool ";
  } else if (module == RES_ADDER) {
    os << "res_adder ";
  } else if (module == PAD) {
    os << "pad ";
  } else {
    os << "Unrecognized ";
  }
  return os.str();
}

void HDFG::Dump() {
  std::ostringstream os;
  os << "\n[HDFG plain order]\n";
  for (auto node : nodes) {
    os << "node[" << node->index << "],";
    os << node->ModuleName();
    os << "->[";
    for (auto succ : node->succ) {
      os << succ->index << " ";
    }
    os << "]\n";
  }
  LOG(INFO) << os.str();
}

/* 
 * bfs starting from last_match
 */
bool HDFG::FindMatch(DFG::Node* node) {
  std::ostringstream os;
  std::unordered_map<Node*, bool> visited;
  for (auto item : nodes) {
    visited[item] = false;
  }
  std::queue<Node*> next_q;
  if (last_match == nullptr) {
    last_match = root;
    next_q.push(last_match);
  } else {
    visited[last_match] = true;
    for (auto item : last_match->succ) {
      next_q.push(item);
    }
  }
  while (!next_q.empty()) {
    Node* current = next_q.front();
    next_q.pop();
    visited[current] = true;
    // check match
    const CallNode* call = static_cast<const CallNode*>(node->ref);
    std::string dfg_opname = call->op.as<OpNode>()->name;
    auto it = op_map.find(dfg_opname);
    if (it != op_map.end()) {
      auto ie = std::find(it->second.begin(), it->second.end(),
        current->module);
      if (ie != it->second.end()) {
        // find a match from tvm op to hardware module
        // os << "node[" << node->index << "] " << call->op;
        // os << " -> " << current->ModuleName() << "\n";
        last_match = current;
        return true;
      }
    }
    // push unvisited successors to queue
    for (auto succ : current->succ) {
      if (!visited[succ]) {
        next_q.push(succ);
      }
    }
  }
  // LOG(INFO) << os.str();
  return false;
}

bool CompareIndex(DFG::Node* a, DFG::Node* b) {
  return a->index > b->index;
}

/* 
 * dfs greedy matching between DFG and HDFG 
 */
void GraphMatchFuser::Partition(DFG dfg) {
  os << "\n[Partition]\n";
  Group* grp = GetLatestGroup(true);
  std::stack<DFG::Node*> Stack;
  std::unordered_map<DFG::Node*, bool> visited;
  Stack.push(dfg.post_dfs_order[0]);
  while (!Stack.empty()) {
    DFG::Node* node = Stack.top();
    Stack.pop();
    // skip visited node (necessary since dfs)
    // handle residual addition case
    if (visited.find(node) != visited.end()) {
      continue;
    }
    visited[node] = true;
    /*for (auto succ : node->succ) {
      if (visited.find(succ) == visited.end())
        Stack.push(succ);
    }*/
    // Push nodes in later topological order into stack first,
    // such that nodes with closer distance to predecessor will
    // be visited before those with farther distance
    // Example : then pad can be grouped with res_add
    //    |
    //  res_add
    //    | \*
    //   pad \*          
    //        \*
    //        /
    //       /
    //    | /
    // concat
    //    |    
    std::vector<DFG::Node*> children;
    for (auto succ : node->succ) {
      if (visited.find(succ) == visited.end())
        children.push_back(succ);
    }
    std::sort(children.begin(), children.end(), CompareIndex);
    for (auto child : children) {
      Stack.push(child);
    }
    if (node->ref->IsInstance<CallNode>()) {
      const CallNode* call = static_cast<const CallNode*>(node->ref);
      std::string opname = call->op.as<OpNode>()->name;
      bool find = hdfg.FindMatch(node);
      auto it = hdfg.op_map.find(opname);
      if (find) {
        // mapped to hardware successfully, add to current group
        grp = GetLatestGroup();
        AddToGroup(node, grp);
        os << "## " << node->index << ": " << call->op
           << " added to group " << grp->index << "\n";
      } else if (it != hdfg.op_map.end()) {
        // primitive operator nodes cannot be mapped to hardware data flow
        // try restart mapping (bfs matching) from source of hardware data flow
        hdfg.last_match = nullptr;
        find = hdfg.FindMatch(node);
        if (find) {
          // successfully mapped after from hardware source
          // add node to a new group
          grp = GetLatestGroup(true);
          AddToGroup(node, grp);
          os << "## " << node->index << ": " << call->op
             << " added to group " << grp->index << "\n";
        } else {
          // restart mapping from hardware source but fail
          // unsuppose to happen
          os << "[ERROR] Cannot map " << call->op
             << " even map from hdfg source\n";
        }
      } else {
        // is CallNode but not on the hardware op_map
        // indicates that it can be virtually mapped to hardware
        // e.g. nn.batch_flatten : alter layout in hardware memory
        // or just change the data fetch pattern without operation
        // under such case, add to current group
        // e.g. expand_dims : ok for now
        grp = GetLatestGroup();
        AddToGroup(node, grp);
        os << "## " << node->index << ": " << call->op
           << " added to group " << grp->index << "\n";
      }
    }
  }
  for(size_t i = 0; i < dfg.post_dfs_order.size(); i++){
    DFG::Node* node = dfg.post_dfs_order[i];
    if (IsOp(node, "add")) {
      if (IsOp(node->pred[0], "nn.pad")) {
        auto grp = gmap_[node->ref];
        gmap_[node->pred[0]->ref] = grp;
        os << "[partial channel fm add] add " << node->index
           << " : update channel expand pad " << node->pred[0]->index
           << " -> group[" << grp->index << "]\n";
        if (IsOp(node->pred[1], "add")) {
          for (auto& item : gmap_) {
            if (item.second == grp) {
              item.second = gmap_[node->pred[1]->ref];
            }
          }
          gmap_[node->pred[1]->ref]->root_ref = grp->root_ref;
        }
      }
    }
  }
  // debug
  os << "====================\n";
  for (size_t i = 0; i < dfg.post_dfs_order.size(); i++) {
    DFG::Node* node = dfg.post_dfs_order[i];
    os << "node[" << i<<"]: ";
    if (node->ref->IsInstance<CallNode>()) {
      CallNode* call = (CallNode*)(node->ref);
      std::string opname = call->op.as<OpNode>()->name;
      os << opname;
    }
    os << "-> group[";
    if (gmap_.find(node->ref) != gmap_.end()) {
      os << gmap_[node->ref]->index;
    } else {
      os << "ERROR";
    }
    os <<"] outputs:[";
    for (auto succ : node->succ) {
      os << succ->index << " "; 
    }
    os << "]\n";
  }
  LOG(INFO) << os.str();
}

bool GraphMatchFuser::IsOp(DFG::Node* node, std::string op_name) {
  if (node->ref->IsInstance<CallNode>()) {
    auto call = reinterpret_cast<const CallNode*>(node->ref);
    if (call->op == Op::Get(op_name)) {
      return true;
    }
  }
  return false;
}

void GraphMatchFuser::AddToGroup(DFG::Node* node, Group* grp) {
  gmap_[node->ref] = grp;
  // update group root with the latest operator node (post dfs order)
  grp->root_ref = node->ref;
  // add all ungrouped preds
  // aim to collect nodes that cannot be mapped via hdfg.op_map
  // e.g. tuple, const, expand_dims
  // counterexample: tuple has 2 relu inputs, which are not considered below,
  // since they should be captured in the Partition() flow
  for (auto pred : node->pred) {
    CollectInputs(pred, grp);
  }
}

/*
 * recursively add inputs of DFG:Node to target group
 * terminate until 
 *   1. a root DFG::Node from other group is met 
 *   2. a grouped DFG::Node is met
 *   3. a ungrouped DFG::Node, which is a CallNode and will be grouped later
 */
void GraphMatchFuser::CollectInputs(DFG::Node* node, Group* grp) {
  auto it = root_map.find(node->ref);
  if (it != root_map.end()) {
    return;
  } else if (gmap_.find(node->ref) != gmap_.end()) {
    return;
  } else {
    if (node->ref->IsInstance<TupleNode>()) {
      os << "tuple\n";
    } else if (node->ref->IsInstance<ConstantNode>()) {
      os << "const\n";
    } else if (node->ref->IsInstance<CallNode>()) {
      const CallNode* call = static_cast<const CallNode*>(node->ref);
      os << call->op << "\n";
      if (call->op.get()->IsInstance<OpNode>()) {
        // Operators in op_map will be taken care of in Partition()
        const OpNode* op = static_cast<const OpNode*>(call->op.get());
        auto ie = hdfg.op_map.find(op->name);
        if (ie != hdfg.op_map.end()) {
          return;
        }
      }
    } else if (node->ref->IsInstance<VarNode>()) {
      os << "var\n";
    } else {
      os << "unknown\n";
    }
    gmap_[node->ref] = grp;
    os << "## " << node->index << ": added to group " << grp->index
       << " via CollectInputs() \n";
  }
  for (auto pred : node->pred) {
    CollectInputs(pred, grp);
  }
}

/*
 * get the old group or create a new one
 */
GraphMatchFuser::Group* GraphMatchFuser::GetLatestGroup(bool create) {
  Group* grp;
  if (create) {
    grp = new Group();
    grp->index = groups.size();
    os << "=====Group" << grp->index << "=====\n";
    groups.push_back(grp);
  } else {
    grp = groups.back();
  }
  // finalize root node for last group
  if (grp->index > 0) {
    Group* last_grp = groups[grp->index-1];
    root_map[last_grp->root_ref] = last_grp;
  }
  return grp;
}


/*
 * copied from fuse_op.cc: utility functions to 
 * encapsulating grouped tvm node corresponding to DFG::Node to function
 */
// Skip primitive function.
Expr GraphMatchFuser::VisitExpr_(const FunctionNode* fn_node) {
  if (fn_node->IsPrimitive()) {
    return GetRef<Expr>(fn_node);
  } else {
    return ExprMutator::VisitExpr_(fn_node);
  }
}

// Transform calls.
Expr GraphMatchFuser::VisitExpr_(const CallNode* call) {
  if (call->op.as<OpNode>()) {
    static auto fnoncomputational =
      Op::GetAttr<TNonComputational>("TNonComputational");

    if (fnoncomputational.get(Downcast<Op>(call->op), false)) {
      return ExprMutator::VisitExpr_(call);
    }

    auto* ret_group = gmap_.at(call);
    Array<Expr> new_args = GetNewArguments(call->args, ret_group);
    // std::cout << call->op << "\n";
    auto new_call = CallNode::make(
        call->op, new_args, call->attrs, call->type_args);

    if (ret_group->root_ref == call) {
      // This is the root of the group
      // create the new call node.
      std::cout << "GROUP[" << ret_group->index << "] ROOT:" << call
                 << " "<< call->op <<"\n";
      return MakeNewFunction(ret_group, call->checked_type(), new_call);
    } else {
      // This is an intermediate node of a fused function
      // simply return the new call.
      return std::move(new_call);
    }
  } else {
    return ExprMutator::VisitExpr_(call);
  }
}

Array<Expr> GraphMatchFuser::GetNewArguments(const tvm::Array<Expr>& args,
  Group* current_group) {
  Array<Expr> new_args;
  for (auto arg : args) {
    auto* arg_group = gmap_.at(arg.get());
    auto type = arg->checked_type();
    Expr new_arg = this->Mutate(arg);
    // std::cout << current_group->index << " " << arg_group->index << ":"
    //           << (bool)(current_group != arg_group) <<"\n";
    if (current_group != arg_group) {
      Var param = ginfo_[current_group].GetOrAllocParam(new_arg, type);
      new_args.push_back(param);
    } else {
      new_args.push_back(new_arg);
    }
  }
  return new_args;
}

Expr GraphMatchFuser::MakeNewFunction(Group* group, Type ret_type, Expr body) {
  // If the function has no call, it is not a primitive function.
  struct HasCallVisitor : ExprVisitor {
    bool has_call = false;
    void VisitExpr_(const CallNode* op) final {
      has_call = true;
    }
  } visitor;
  visitor(body);
  const GroupInfo& ginfo = ginfo_[group];
  auto func = FunctionNode::make(ginfo.params, body, ret_type, {});
  // Set function primitive based on if has sub routine
  // we set function not primitive here since we need to visit 
  // in the later passes
  visitor.has_call = false;
  func =
    FunctionSetAttr(func, attr::kPrimitive, tvm::Integer(visitor.has_call));
  auto ret = CallNode::make(func, ginfo.arguments, Attrs());
  CHECK(WellFormed(ret)) << AsText(ret, false);
  return ret;
}

Expr GraphMatchFuser::VisitExpr_(const TupleNode* tuple) {
  if (gmap_.find(tuple) != gmap_.end()) {
    auto* ret_group = gmap_.at(tuple);
    if (ret_group->root_ref == tuple) {
      return ExprMutator::VisitExpr_(tuple);
    }
    // This tuple is an intermediate node in the group
    Array<Expr> new_fields = GetNewArguments(tuple->fields, ret_group);
    return TupleNode::make(new_fields);
  } else {
    // If model has multiple outputs, top function gets a tuple as output
    // e.g. yolov3
    // In such case, we don't make new functions so output tuple is left ungrouped    
    return ExprMutator::VisitExpr_(tuple);
  }
}

Expr GraphMatchFuser::VisitExpr_(const TupleGetItemNode* tuple_get) {
  auto* ret_group = gmap_.at(tuple_get);
  auto new_tuple = GetNewArguments({tuple_get->tuple}, ret_group)[0];
  auto new_node = TupleGetItemNode::make(new_tuple, tuple_get->index);
  if (ret_group->root_ref == tuple_get) {
    if (gmap_.at(tuple_get->tuple.get()) != ret_group) {
      // Isolated. This case occurs when tuple is created by an Opaque op
      // e.g. multibox_transform_loc
      return ExprMutator::VisitExpr_(tuple_get);
    }
    // A new function whose output is a tuple field access
    return MakeNewFunction(ret_group, tuple_get->checked_type(), new_node);
  }
  // This is an intermediate node in the group
  return std::move(new_node);
}

Expr GraphMatchFuser::Transform(const Expr& body) {
  return this->Mutate(body);
}

/*
 * pass top function
 */
Expr FuseOP(const Expr& expr, const IRModule& module) {
  // create dfg and hdfg
  DFG dfg = DFG::Creator().Prepare(expr);
  // dfg.Dump();
  HDFG hdfg = HDFG::Create();
  // hdfg.Dump();
  GraphMatchFuser gmf = GraphMatchFuser(hdfg);
  // group nodes
  gmf.Partition(dfg);
  // encapsulate grouped to one function
  Expr ret = gmf.Transform(expr);
  return ret;
}

namespace transform {

Pass FuseOpHWCustom() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>
    pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(FuseOP(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "FuseOpHWCustom",
                            {ir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseOpHWCustom")
.set_body_typed(FuseOpHW);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
