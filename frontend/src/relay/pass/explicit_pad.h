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
#ifndef SRC_RELAY_PASS_EXPLICIT_PAD_H_
#define SRC_RELAY_PASS_EXPLICIT_PAD_H_
/*
 * extract implicit padding of conv/pool operator,
 * for better hardware mapping with post padding scheme
 *
 * OPU performs padding by writing specific data when writing 
 * to ddr. e.g. write extra zeros for post zero padding
 *
 * in the example below, after padding being explicit, compiler 
 * can recognize pad=[1,1,1,1] as the post padding after relu,
 * which increases the opportunity for following operator fusion
 *
 *       relu                        relu
 *         |                           |
 *         |                           |
 *      conv2d(pad=[1,1,1,1])  ->  pad=[1,1,1,1]
 *                                     |
 *                                     | 
 *                            conv2d(pad=[0,0,0,0])  
 */ 

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <string>

#include "./pattern_util.h"
#include "./hw_info.h"

namespace tvm {
namespace relay {

// traverse expr and collect implicit padding
class ImplicitPadCollector : private ExprVisitor{
 public:
    // operator (with implicit padding) - padding size
    std::unordered_map<const CallNode*, Array<Array<PrimExpr>>>
      pad_node_size_map;

    void Prepare(const Expr& body);

 private:
    std::ostringstream os;

    void VisitExpr_(const CallNode* call);
    // utility function to check if padding is implicit (non-zero padding size)
    bool CheckImplicitPad(Array<PrimExpr> padding);

    // conv_attrs->padding:[1,1,1,1]
    // -> pad_attrs->pad_width:[[0,0],[0,0],[1,1],[1,1]]
    // NCHW layout
    Array<Array<PrimExpr>>
      PadTransform(Array<PrimExpr> padding, std::string layout);
};


// mutate expr by explicitly insert non-zero padding (implicit previously)
class GraphMutatorPad : public ExprMutator {
 public:
    std::ostringstream os;
    std::unordered_map<const CallNode*, Array<Array<PrimExpr>>>
      pad_node_size_map;

    // Run the transform
    Expr Transform(const Expr& body);

    Expr VisitExpr_(const CallNode* call);

    // insert pad with pad_width before expr data
    Expr InsertPad(Expr data, Array<Array<PrimExpr>> pad_width);

    // create zero padding array,
    // used to replace previously implicit non-zero padding
    Array<IndexExpr> MakeZeroPadding(size_t size);
};

}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_EXPLICIT_PAD_H_
