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
#include "../../src/relay/pass/fuse_ops_hw.h"
#include <tvm/top/operation.h>

TVM_REGISTER_GLOBAL("schedule")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      *rv = topi::generic::schedule_injective(args[0], args[1]);
    });

TEST(Relay, FuseOpsHW) {
  using namespace tvm;
  // Create a function for optimization.
  auto tensor_type = relay::TensorTypeNode::make({1, 1, 4, 4}, DataType::Float(32));
  auto kernel_type = relay::TensorTypeNode::make({3, 3, 1, 1}, DataType::Float(32));
  auto x = relay::VarNode::make("x", tensor_type);
  auto w = relay::VarNode::make("w", kernel_type);
  auto strides = {relay::IndexExpr(1), relay::IndexExpr(1)};
  auto padding = {relay::IndexExpr(0), relay::IndexExpr(0),
                       relay::IndexExpr(0), relay::IndexExpr(0)};
  auto dilation = {relay::IndexExpr(1), relay::IndexExpr(1)};
  int groups = 1;
  auto channels = relay::IndexExpr(1);
  auto kernel_size = {relay::IndexExpr(3), relay::IndexExpr(3)};
  std::string data_layout = "NCHW";
  std::string kernel_layout = "HWIO";
  std::string out_layout = data_layout;
  auto out_dtype = runtime::DataType::Float(32);
  auto conv = relay::Conv2D(x, w, strides, padding, dilation, groups,
                         channels, kernel_size, data_layout, kernel_layout,
                         out_layout, out_dtype);
  auto b_type = relay::TensorTypeNode::make({1, 1, 1, 1}, DataType::Float(32));
  auto b = relay::VarNode::make("b", b_type);
  auto add = relay::Add(conv, b);
  auto relu = relay::Relu(add);
  auto pointwise_kernel_type = relay::TensorTypeNode::make({1, 1, 1, 4}, DataType::Float(32));
  auto wp = relay::VarNode::make("wp", pointwise_kernel_type);
  auto kernel_size_wp = {relay::IndexExpr(1), relay::IndexExpr(1)};
  auto pointwise_conv = relay::Conv2D(relu, wp, strides, padding, dilation, groups,
                         4, kernel_size_wp, data_layout, kernel_layout,
                         out_layout, out_dtype);
  auto bp_type = relay::TensorTypeNode::make({1, 4, 1, 1}, DataType::Float(32));
  auto bp = relay::VarNode::make("bp", bp_type);
  auto z = relay::Add(pointwise_conv, bp);
  relay::Function func =
      relay::FunctionNode::make(relay::FreeVars(z), z, relay::Type(), {});


  tvm::Array<relay::transform::Pass> pass_seqs{
      relay::transform::InferType(),
      relay::transform::PrintIR(false),
      relay::transform::FuseOpHWCustom(),
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
  // CHECK(relay::AlphaEqual(f, expected)) 
  // cannot work for function within function for now
  // need to do further verification
  
  /* 
   * Example test for DFG::Prepare()
   */
  relay::DFG DFGTest;
  relay::DFG::Creator CT;
  auto mod_test = IRModule::FromExpr(func);
  auto pass_ctx_test = relay::transform::PassContext::Create();
  pass_ctx_test->opt_level = 1;
  pass_ctx_test->fallback_device = 1;
  {
    tvm::With<relay::transform::PassContext> ctx_scope(pass_ctx);
    tvm::With<tvm::Target> tctx(tvm::Target::Create("llvm"));
    mod = relay::transform::InferType()(mod);
  }
  relay::Function f_test = Downcast<relay::Function>(mod_test->Lookup("main"));
  DFGTest = CT.Prepare(f_test);
  DFGTest.Dump();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}