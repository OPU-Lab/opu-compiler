#include <tvm/relay/analysis.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm.h>
#include <tvm/relay/expr.h>
#include <memory>

#include "utils.h"

namespace tvm{
namespace relay{
namespace backend{
    
using TargetsMap = Map<tvm::Integer, tvm::Target>;
  /*!
 * \brief Output of building module
 *
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen {
 public:
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphRuntimeCodegen");
    mod = (*pf)();
  }
  ~GraphCodegen() {}

  void Init(runtime::Module* m, TargetsMap targets) {
    CallFunc("init", m, targets);
  }

  void Codegen(const Function& func) {
    CallFunc("codegen", func);
  }

  std::string GetJSON() {
    return CallFunc<std::string>("get_graph_json", nullptr);
  }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module> >("get_external_modules", nullptr);
  }

  Map<std::string, Array<LoweredFunc> > GetLoweredFunc() {
    return CallFunc<Map<std::string, Array<LoweredFunc> > >("get_lowered_funcs", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<tvm::PrimExpr> >("list_params_name", nullptr);
    for (auto expr : names) {
      auto key = expr.as<ir::StringImmNode>()->value;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

 protected:
  tvm::runtime::Module mod;
  template<typename R, typename ...Args>
  R CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template<typename ...Args>
  void CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};
  
}    
}    
}