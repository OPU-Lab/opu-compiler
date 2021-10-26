#include <gtest/gtest.h>
#include <iostream>
#include <topi/generic/injective.h>
#include <tvm/build_module.h>
#include <tvm/packed_func_ext.h>
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/attrs/nn.h>
#include "../src/relay/pass/pattern_util.h"
#include <tvm/top/operation.h>
#include "../src/relay/pass/fuse_op_hw.h"

#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/expr.h>

TVM_REGISTER_GLOBAL("schedule")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      *rv = topi::generic::schedule_injective(args[0], args[1]);
    });


// TEST(Relay, Update) {
  // OpPatternKind op_pattern = kElemWise;
  // IndexedGraph::Node* parent = NULL;
  // IndexedGraph::Creator::Update();
  // relay::IndexedGraph::Creator::Prepare(NULL);

// }

TEST(Relay, collectGroupNodes){
  using namespace tvm;

  relay::Partitioner ParTest;
  relay::IndexedGraph::Node NodeIn;
  relay::Partitioner::Group GroupIn;

  relay::Attr attrIn("var");
  NodeIn.attr = &attrIn;
  // NodeIn.inputs = 
  support::LinkedList<relay::IndexedGraph::Edge> inputsIn;


  ParTest.collectGroupNodes(&NodeIn, &GroupIn);
}

TEST(Relay, check){
  using namespace tvm;

  relay::Partitioner ParTest;
  relay::IndexedGraph::Node NodeRef_1;
  relay::IndexedGraph::Node NodeIn_1;

  relay::Attr attrIn("var");
  NodeIn_1.attr = &attrIn;
  ParTest.check(&NodeRef_1, &NodeIn_1);

  relay::IndexedGraph::Node NodeRef_2;
  relay::IndexedGraph::Node NodeIn_2;

  relay::Attr attrIn_2("var");
  NodeIn_2.attr = &attrIn_2;
  ParTest.check(&NodeRef_2, &NodeIn_2);
}

TEST(Relay, OpFuser_VisitExpr_){
  using namespace tvm;

  relay::FunctionNode FNodeIn_1;
  relay::OpFuser VisitExprTest_1;
  FNodeIn_1.ret_type = Type();
  FNodeIn_1.body = RelayExpr();
  VisitExprTest_1.VisitExpr_(&FNodeIn_1);

}
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
