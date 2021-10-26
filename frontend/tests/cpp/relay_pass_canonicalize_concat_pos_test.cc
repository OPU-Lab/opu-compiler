#include <gtest/gtest.h>
#include <iostream>
#include <topi/generic/injective.h>
#include <tvm/build_module.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/expr.h>
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/attrs/nn.h>
#include "../../src/relay/pass/pattern_util.h"
#include <tvm/top/operation.h>

TVM_REGISTER_GLOBAL("schedule")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      *rv = topi::generic::schedule_injective(args[0], args[1]);
    });

TEST(Relay, CanonicalizeConcatPos) {
  // tuple inputs can only be CallNode instead of ConcatNode for now
  using namespace tvm;
  auto tensor_type = relay::TensorTypeNode::make({1, 4, 4, 1}, DataType::Float(32));

  // Create a function for optimization.
  auto a = relay::VarNode::make("a", tensor_type);
  auto x = relay::VarNode::make("x", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto a0 = relay::CallNode::make(add_op, {a, a});
  auto x0 = relay::CallNode::make(add_op, {x, x});
  auto concat_op = relay::Op::Get("concatenate");
  auto t = relay::TupleNode::make({a0, x0});
  auto y = relay::MakeConcatenate(t ,3); 
  auto pool_size = Array<relay::IndexExpr>{relay::IndexExpr(2),relay::IndexExpr(2)};
  auto strides = Array<relay::IndexExpr>{relay::IndexExpr(2),relay::IndexExpr(2)};
  auto padding = Array<relay::IndexExpr>{relay::IndexExpr(0),relay::IndexExpr(0)};
  auto z = relay::AvgPool2D(y, pool_size, strides, padding, "NHWC", false, false);
  relay::Function func =
      relay::FunctionNode::make(relay::FreeVars(z), z, relay::Type(), {});


  tvm::Array<relay::transform::Pass> pass_seqs{
      relay::transform::InferType(),
      relay::transform::PrintIR(false),
      relay::transform::CanonicalizeConcatPos(),
      relay::transform::PrintIR(false)
  };
  relay::transform::Pass seq = relay::transform::Sequential(pass_seqs);
  auto mod = IRModule::FromExpr(func);
  auto pass_ctx = relay::transform::PassContext::Create();
  pass_ctx->opt_level = 1;
  pass_ctx->fallback_device = 1;
  {
    tvm::With<relay::transform::PassContext> ctx_scope(pass_ctx);
    tvm::With<tvm::Target> tctx(tvm::Target::Create("llvm"));
    mod = seq(mod);
  }
  CHECK(mod.defined());
  auto entry_func = mod->GetGlobalVar("main");
  CHECK(entry_func.defined());
  relay::Function f = Downcast<relay::Function>(mod->Lookup("main"));
  CHECK(f.defined());

  // Expected function
  auto a1 = relay::VarNode::make("a", tensor_type);
  auto x1 = relay::VarNode::make("x", tensor_type);
  auto a11 = relay::CallNode::make(add_op, {a1, a1});
  auto x11 = relay::CallNode::make(add_op, {x1, x1});
  auto a12 = relay::AvgPool2D(a11, pool_size, strides, padding, "NHWC", false, false);
  auto x12 = relay::AvgPool2D(x11, pool_size, strides, padding, "NHWC", false, false);
  auto t1 = relay::TupleNode::make({a12, x12});
  auto y1 = relay::MakeConcatenate(t1 ,3); 
  relay::Function expected_func =
      relay::FunctionNode::make(relay::FreeVars(y1), y1, relay::Type(), {});

  // Infer type for the expected function.
  auto mod1 = IRModule::FromExpr(expected_func);
  mod1 = relay::transform::InferType()(mod1);
  auto expected = mod1->Lookup("main");
  CHECK(relay::AlphaEqual(f, expected));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
