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

/*!
 *
 * \file src/tvm/relay/pass/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/ir/attrs.h>
#include "./pattern_util.h"
#include "../../support/arena.h"
#include "./hw_info.h"


namespace tvm {
namespace relay {
using support::LinkNode;
using support::LinkedList;
class Attr{
 public:
   std::string type{""};
   bool mapped{false};
   Attr(std::string type){
     this->type = type; 
   };  
   std::string mapped_name{""}; 
   const Attrs* op_attr{nullptr};
   Array<PrimExpr> shape;
};

class IndexedGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
    Attr* attr;
    LinkedList<Edge> inputs;
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], "
         << GetRef<ObjectRef>(node->ref)
         << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static IndexedGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

class IndexedGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena)
      : arena_(arena) {}

  IndexedGraph Prepare(const Expr& body);

 private:
  std::ostringstream os;
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  IndexedGraph graph_;
  // attribute equal comparator
  AttrsEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, IndexedGraph::Node* parent, OpPatternKind pattern);

  void AddNode(const tvm::Object* key, Attr* attr);

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

/*
struct Edge {
    Node* node{nullptr};
    OpPatternKind pattern{kOpaque};
  };
struct Node {
    const tvm::Object* ref{nullptr};
    size_t index{0};
    bool extern_ref{false};
    OpPatternKind pattern{kOpaque};
    LinkedList<Edge> outputs;
    Attr* attr;
  };
*/
class Partitioner{
 public:
  struct Group{
    std::vector<IndexedGraph::Node*> nodes;
    IndexedGraph::Node* root;
    std::vector<IndexedGraph::Node*> nodes_i;
    std::vector<Group*> parents;
    std::vector<Group*> children;
    const Object* root_ref{nullptr};
    size_t index;
  };
  std::vector<Group*> groups;
  
  support::Arena* arena_;
  IndexedGraph::Node* HwSrc;
  IndexedGraph::Node* HwSrc_t;
  std::vector<std::pair<IndexedGraph::Node*, IndexedGraph::Node*>> mapped;
  std::vector<IndexedGraph::Node*> roots;
  std::unordered_map<std::string, std::vector<std::string>> OpMap = {
    {"tuple",{"tuple"}},
    {"concatenate",{"concat"}},
    
    {"add",{"pre_adder","post_adder","res_adder"}},
    {"multiply",{"pre_multiplier","post_multiplier","se_block"}},
    
    {"nn.conv2d",{"ipa"}},
    {"nn.dense",{"ipa"}},
    
    {"nn.max_pool2d",{"pool"}},
    {"nn.avg_pool2d",{"pool"}},
    {"nn.global_avg_pool2d",{"pool"}},
    
    {"nn.relu",{"pre_comparator","post_comparator"}},
    {"nn.leaky_relu",{"pre_comparator","post_comparator"}}
  };
  std::ostringstream os;
  
  IndexedGraph::Node* InitNode(std::string op_type, IndexedGraph::Node* src, size_t index);
  
  void InitHwGraph();
  
  void ResetHwGraph();
  
  // dfs collect node to group->nodes, terminate when encounter another group root
  void collectGroupNodes(IndexedGraph::Node* src, Group* group);
  
  void dump(IndexedGraph::Node* node);
  
  IndexedGraph::Node* check(IndexedGraph::Node* ref, IndexedGraph::Node* node);
  
  std::vector<Group*> match(support::Arena* arena, std::vector<IndexedGraph::Node*> graph);
};


class OpFuser: private ExprMutator {
 public:
  support::Arena arena_;
  std::ostringstream os;
  
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
    // OPU information
    OpuInfo* info{nullptr};
  };
  std::unordered_map<const Object*, Partitioner::Group*> gmap_;
  std::unordered_map<Partitioner::Group*, GroupInfo> tginfo_;
  OpuInfoCollection oic;
  
  // Run the transform
  Expr Transform(const Expr& body);
  
  void Update(std::vector<Partitioner::Group*> groups);
  
  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node);

  // Transform calls.
  Expr VisitExpr_(const CallNode* call);
  
  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args, Partitioner::Group* current_group);

  Expr MakeNewFunction(Partitioner::Group* group, Type ret_type, Expr body);
  
  Expr VisitExpr_(const TupleNode* tuple);

  Expr VisitExpr_(const TupleGetItemNode* tuple_get);
};

}  // namespace relay
}  // namespace tvm
