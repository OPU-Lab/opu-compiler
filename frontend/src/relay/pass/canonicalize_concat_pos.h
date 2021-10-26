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
#ifndef SRC_RELAY_PASS_CANONICALIZE_CONCAT_POS_H_
#define SRC_RELAY_PASS_CANONICALIZE_CONCAT_POS_H_
/*
 * Move channel-wise operator successors after concatenation to be 
 * predecessors of concatenation.
 *
 * OPU hardware preforms concat by fetching data from different ddr 
 * addresses to the same on-chip bram, which only happens at the start
 * of hardware dataflow, before IPA. So this pass makes sure concat 
 * appear before IPA(conv/fc).
 *
 *           relu   relu    relu      relu
 *             \    /        \        /
 *              \  /          \      /
 *             concat        pool  pool  
 *               |      -->    \   /
 *               |              \ /
 *              pool           concat
 *               |               |  
 *               |               |
 *             conv2d          conv2d 
 */

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <vector>
#include <unordered_map>

#include "./pattern_util.h"
#include "../../support/arena.h"

namespace tvm {
namespace relay {

// traverse expr and collect concat node and its predecessors
class NodeCollector : private ExprVisitor{
 public:
  // concat - (tuple inputs)
  std::unordered_map<const CallNode*, std::vector<const CallNode*>>
    concat_inputs_map;
  // tuple input - expr
  std::unordered_map<const CallNode*, Expr> expr_map;
  // direct successor (which we want to move before concat) - concat
  std::unordered_map<const CallNode*, const CallNode*> succ_concat_map;

  std::unordered_map<const tvm::Object*, bool> memo_;
  
  void Prepare(const Expr& body);

 private:
  std::ostringstream os;

  void VisitExpr_(const TupleNode* op);

  void VisitExpr_(const CallNode* call);

  // channel-wise successor after concat can be moved
  bool IsChannelWise(const CallNode* call);
};


// mutate expr by moving concat's direct successors
class GraphMutator : public ExprMutator {
 public:
  std::ostringstream os;
  std::unordered_map<const CallNode*, std::vector<const CallNode*>>
    concat_inputs_map;
  std::unordered_map<const CallNode*, const CallNode*> succ_concat_map;

  // Run the transform
  Expr Transform(const Expr& body);

  Expr VisitExpr_(const CallNode* call);
};


}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_CANONICALIZE_CONCAT_POS_H_
