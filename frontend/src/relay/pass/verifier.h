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
#include <tvm/top/operation.h>

#include <stdlib.h>
#include <time.h>

#include "./pattern_util.h"
#include "nlohmann/json.hpp"
#include "./hw_info.h"
#include "../backend/graph_codegen_wrapper.h"
#include "../../runtime/graph/graph_runtime.h"
#include "cnpy.h"

using json = nlohmann::json;
namespace tvm {
namespace relay {

class Verifier {
 public:
  bool RunTest(Expr& ref);
  bool EqShape(tvm::runtime::NDArray A, tvm::runtime::NDArray B);
  std::vector<std::pair<size_t, size_t>> FindMatch(
    std::vector<tvm::runtime::NDArray> vec_a,
    std::vector<tvm::runtime::NDArray> vec_b);
  class Deployer;
  class Creator;
};

/*
 * deploy expr on target device e.g. cpu
 */
class Verifier::Deployer {
 public:
  DLContext ctx {kDLCPU, 0};
  DLDataType dtype {kDLFloat, 32, 1};
  std::vector<tvm::runtime::NDArray> results;
  
  void Run(Expr& body, std::vector<tvm::runtime::NDArray> input_ndarrays, bool raw = false);
  std::string GetLayout(Expr& body);
  std::vector<int64_t> GetInputShape(Expr& body);
  // Get tensor size
  int64_t GetTensorSize(runtime::NDArray data);
  std::vector<tvm::runtime::NDArray> GetInputs(std::vector<int64_t> input_shape);
  std::vector<Expr> GetOutputs(Expr& body);
  
  bool IsEqual(tvm::runtime::NDArray A, tvm::runtime::NDArray B);
};

/*
 * construct graph from OPU_IR.json
 */
class Verifier::Creator {
 public:
  DLContext ctx {kDLCPU, 0};
  DLDataType dtype {kDLFloat, 32, 1};
  std::string target_layout {"NCHW"};
  int concat_axis = 1;
  std::unordered_map<int, OpuInfo*> info_map_;
  // layer index - Expr
  std::unordered_map<int, Expr> emap_;
  std::vector<Expr> outputs;
 
  void Prepare();
  void BuildLayer(OpuInfo* info);
  void Update(int index, Expr expr);
  std::vector<IndexExpr> GetIndexExprVec(std::vector<int64_t> ishape);
  Expr MakePad(Expr data, std::vector<int64_t> padding_size);
  Expr MakeDePad(Expr data, std::vector<int64_t> padding_size);
  bool IsEmpty(std::vector<int64_t> vec);
  Constant LoadConstant(int index, std::string name, std::vector<int64_t> shape);
  IndexExpr ToIndexExpr(int64_t data) {
    return IndexExpr(static_cast<int>(data));
  }
  Expr MakeActivation(Expr data, int type);
  Expr MakePooling(Expr data, int type, std::vector<int64_t> pooling_size,
    std::vector<int64_t> pooling_stride, std::vector<int64_t> pool_padding_size);
  Expr MakeResidualAdd(Expr data, int has_residual, std::vector<int64_t> residual_source);
  Expr MakeUpsampling(Expr data, int scale, std::string method);
};

}  // namespace relay
}  // namespace tvm