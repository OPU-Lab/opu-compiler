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
#ifndef SRC_RELAY_PASS_FUSE_OPS_HW_H_
#define SRC_RELAY_PASS_FUSE_OPS_HW_H_
/*
 * fuse operators (minize off-chip communication) according to OPU
 * data flow. 
 * 
 * first build a data flow graph from input model (class DFG) and build
 * a data flow graph for custom target hardware (class HDFG). Then match 
 * node from DFG to HDFG with operator mapping scheme (HDFG::op_map) and 
 * greedy algorithm. Fused operators are encapsulated to a tvm function as 
 * mutated expr output.
 *
 *      concat                
 *        |                fused(concat--conv2d--relu)
 *      conv2d                    |
 *        |         ->            |
 *       relu              fused(conv2d--pool)
 *        |
 *      conv2d
 *        |
 *       pool 
 */

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/ir/attrs.h>

#include <vector>
#include <unordered_map>
#include <queue>
#include <stack>
#include <string>

#include "./pattern_util.h"
#include "../../support/arena.h"
#include "./hw_info.h"

namespace tvm {
namespace relay {

// data flow graph from input expr
class DFG {
 public:
  // dfg node structure
  struct Node {
    const tvm::Object* ref{nullptr};
    size_t index{0};
    std::vector<Node*> pred;
    std::vector<Node*> succ;
    // element-wise(residual add between two fms) or channel-wise(bias)
    bool isEW{false};
  };
  // The node map that maps tvm node to DFG::Node
  std::unordered_map<const tvm::Object*, Node*> node_map;
  // All the nodes in post DFS order
  std::vector<Node*> post_dfs_order;
  // for debug
  void Dump();
  // traverse expr and build dfg
  class Creator;
};

class DFG::Creator : private ExprVisitor {
 public:
  DFG Prepare(const Expr& body);
  // update node inputs with collected outputs
  void AnnotateInputs();

 private:
  std::ostringstream os;
  DFG graph_;
  // Update the message stored at the node.
  void Update(const Expr& node, DFG::Node* parent);

  void AddNode(const tvm::Object* key);

  // Post order tree
  void VisitExpr_(const FunctionNode* op);

  void VisitExpr_(const ConstantNode* op);

  void VisitExpr_(const CallNode* call);

  void VisitExpr_(const TupleNode* op);

  void VisitExpr_(const TupleGetItemNode* op);

  void VisitExpr_(const VarNode* op);

  void VisitExpr_(const LetNode* op);

  void VisitExpr_(const IfNode* op);

  void VisitExpr_(const RefCreateNode* op);

  void VisitExpr_(const RefReadNode* op);

  void VisitExpr_(const RefWriteNode* op);

  void VisitExpr_(const MatchNode* op);
};

// hardware data flow graph
class HDFG {
 public:
  // hardware module names
  enum HWMODULE {
    MEM_RD,
    IPA,
    COMPARATOR,
    POOL,
    ADDER,
    MULTIPLIER,
    RES_ADDER,
    PAD,
    UPSAMPLE,
    SE,
    MEM_WR,
    NOP
  };
  struct Node {
    HWMODULE module;
    std::vector<Node*> succ;
    size_t index;
    std::string ModuleName();
  };
  // start of hdfg
  Node* root;
  std::vector<Node*> nodes;
  // utility function building hdfg
  // create from some description file later
  static HDFG Create();
  Node* AddNode(HWMODULE module);
  void LinkFromTo(Node* current, Node* pred);
  // debug print
  void Dump();
  // operator mapping
  std::unordered_map<std::string, std::vector<HWMODULE>> op_map {
        {"concatenate", {MEM_RD}},
        {"add", {ADDER, RES_ADDER}},
        {"multiply", {MULTIPLIER, SE}},
        {"nn.conv2d", {IPA}},
        {"nn.conv2d_transpose", {IPA}},
        {"nn.dense", {IPA}},
        {"nn.max_pool2d", {POOL}},
        {"nn.avg_pool2d", {POOL}},
        {"nn.global_avg_pool2d", {POOL}},
        {"nn.relu", {COMPARATOR}},
        {"nn.leaky_relu", {COMPARATOR}},
        {"nn.pad", {PAD}},
        {"nn.upsampling", {UPSAMPLE}}
  };
  // bookkeep traversal position in hdfg
  Node* last_match{nullptr};
  // check if a DFG::Node can be mapped to hdfg based on last_match
  bool FindMatch(DFG::Node* node);
};

// mutate expr by capsulating fused operators to functions
class GraphMatchFuser : private ExprMutator {
 public:
  explicit GraphMatchFuser(HDFG hdfg) {
    this->hdfg = hdfg;
  }
  HDFG hdfg;
  std::ostringstream os;

  // group related (group===fused operators)
  struct Group {
   public:
    std::vector<DFG::Node*> nodes;
    const tvm::Object* root_ref{nullptr};
    size_t index;
  };
  bool IsOp(DFG::Node* node, std::string op_name);
  std::unordered_map<DFG::Node*, bool> channel_pad_nodes;
  void AddToGroup(DFG::Node* node, Group* grp);
  void CollectInputs(DFG::Node* node, Group* grp);
  std::vector<Group*> groups;
  Group* GetLatestGroup(bool create = false);
  std::unordered_map<const tvm::Object*, Group*> root_map;

  // copied from tvm original fuse_ops.cc
  // utility functions creating function for fused operators
  struct GroupInfo {
   public:
    // The parameters of the function.
    Array<Var> params;
    // The arguments to call the functions.
    Array<Expr> arguments;
    // Get a new parameter or allocate an old one
    Var GetOrAllocParam(const Expr& expr, const Type& type) {
      // run linear scan as most fused groups contain only a few inputs.
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (expr.same_as(arguments[i])) return params[i];
      }
      // create a new parameter.
      std::ostringstream os;
      os << "p" << params.size();
      auto var = VarNode::make(os.str(), type);
      params.push_back(var);
      arguments.push_back(expr);
      return var;
    }
  };
  std::unordered_map<const tvm::Object*, Group*> gmap_;
  std::unordered_map<Group*, GroupInfo> ginfo_;

  void Partition(DFG dfg);


  Expr Transform(const Expr& body);
  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node);

  // Transform calls.
  Expr VisitExpr_(const CallNode* call);

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args,
    Group* current_group);

  Expr MakeNewFunction(Group* group, Type ret_type, Expr body);

  Expr VisitExpr_(const TupleNode* tuple);

  Expr VisitExpr_(const TupleGetItemNode* tuple_get);
};

}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_FUSE_OPS_HW_H_
