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
#include "./quantize.h"
#include "../backend/graph_codegen_wrapper.h"
#include "../../runtime/graph/graph_runtime.h"
#include "cnpy.h"

namespace tvm {
namespace relay {

/*
 * Top function
 */
void QNN::Prepare(const Expr& body) {
  os << "\n[Quantize]\n";
  // Quantize constant in the traversal
  this->VisitExpr(body);
  // LOG(INFO) << os.str();
  QCC qcc = QCC(funcs.size());
  qcc.Prepare(funcs);
  // Set model input
  tvm::runtime::NDArray A =
    tvm::runtime::NDArray::Empty(input_shape, dtype, ctx);
  cnpy::NpyArray arr = cnpy::npy_load("input.npy");
  std::vector<float> loaded_data = arr.as_vec<float>();
  float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
  std::memcpy(data, &loaded_data[0], sizeof(float)*GetTensorSize(A));
  output_dict[0] = A;
  // Run sub-functions in topological order
  for (size_t i = 0; i < funcs.size(); i++) {
    Func* func = funcs[i];
    os << "layer[" << func->info->index << "]\n";
    const FunctionNode* fn = static_cast<const FunctionNode*>(func->ref);
    Expr re = GetRef<Expr>(fn);
    // Get input NDArrays
    std::vector<tvm::runtime::NDArray> input_ndarrays;
    for (auto input_index : func->info->input_layer) {  // concat inputs
      input_ndarrays.push_back(output_dict[input_index]);
    }
    // Quantize inputs
    int ifl = SearchArrayFraclen(input_ndarrays);
    os << "  input <" << ifl << "/ 8>\n";
    func->info->quantized = true;
    qcc.SetInput(func->info->index - 1, ifl);
    // Compute output
    auto ndarray = Compute(re, input_ndarrays);
    output_dict[func->info->index] = ndarray;
    // Quantize final outputs
    auto it = std::find(
        func->info->output_layer.begin(),
        func->info->output_layer.end(),
        -1);
    if ((func->info->output_layer.size() == 1) &&
        (it != func->info->output_layer.end())) {
      int ofl = SearchFraclen(ndarray);
      os << "  output <" << ofl << "/ 8>\n";
      func->info->quantized = true;
      qcc.SetOutput(func->info->index - 1, ofl);
    }
  }
  qcc.Apply(funcs);
  LOG(INFO) << os.str();
  // Code for argmax prediction
  /*
  auto Y = output_dict[funcs.size()];
  auto y_iter = static_cast<float*>(Y.ToDLPack()->dl_tensor.data);
  auto max_iter = std::max_element(y_iter, y_iter + 1000);
  auto max_index = std::distance(y_iter, max_iter);
  std::cout << "The maximum position in output vector is: " << max_index << std::endl;
  */
}

int QNN::QCC::GetInput(int layer_index) {
  return layer_index * 2;
}

int QNN::QCC::GetOutput(int layer_index) {
  return layer_index * 2 + 1;
}

int QNN::QCC::Find(int index) {
  if (parent[index] == -1) {
    return index;
  }
  return Find(parent[index]);
}

void QNN::QCC::Union(int x, int y) {
  int xset = Find(x);  
  int yset = Find(y);  
  if (xset != yset) {  
    parent[xset] = yset;  
  }  
}

void QNN::QCC::Dump() {
  for (auto item : gmap_) {
    os << "===== [QCC] Group " << item.first << "=====\n";
    for (auto x : item.second) {
      int residue = x % 2;
      int layer_index = (x - x % 2) / 2 + 1;
      os << "layer[" << layer_index <<"](";
      int index;
      if (residue == 1) {
        os << "out";
        index = GetOutput(layer_index);
      } else {
        os << "in";
        index = GetInput(layer_index);
      }
      os << ")";
      auto it = vmap_.find(Find(index));
      if (it == vmap_.end()) {
        os << " ERROR";
      } else {
        os << " " << it->second;
      }
      os << "\n";
    }
  }
}

void QNN::QCC::Prepare(std::vector<Func*> funcs) {
  for (auto func : funcs) {
    int layer_index = func->info->index - 1;
    for (auto x : func->info->input_layer) {
      auto it = std::find(func->info->residual_source.begin(),
                          func->info->residual_source.end(),
                          x);
      if (x > 0) {
        if (it != func->info->residual_source.end()) {
          os << "union " << func->info->index << "'s out, " << x << "'s out\n";
          int a = Find(GetOutput(layer_index));
          int b = Find(GetOutput(x - 1));
          if (a != b) {
            Union(a, b);
          }
        } else {
          os << "union " << func->info->index << "'s in, " << x << "'s out\n";
          int a = Find(GetInput(layer_index));
          int b = Find(GetOutput(x - 1));
          if (a != b) {
            Union(a, b);
          }
        }
      }
    }
  }
  for (size_t i = 0; i < size; i++) {
    int ii = static_cast<int>(i);
    gmap_[Find(ii)].push_back(ii); 
  }
}

void QNN::QCC::SetInput(int layer_index, int value) {
  int index = GetInput(layer_index);
  int gid = Find(index);
  if (vmap_.find(gid) == vmap_.end()) {
    vmap_[gid] = value;
  }
}

void QNN::QCC::SetOutput(int layer_index, int value) {
  int index = GetOutput(layer_index);
  int gid = Find(index);
  if (vmap_.find(gid) == vmap_.end()) {
    vmap_[gid] = value;
  }
}

void QNN::QCC::Apply(std::vector<Func*> funcs) {
  for (auto func : funcs) {
    int layer_index = func->info->index - 1;
    int ifl = vmap_.at(Find(GetInput(layer_index)));
    func->info->input_fraclen = ifl;
    int ofl = vmap_.at(Find(GetOutput(layer_index)));
    func->info->output_fraclen = ofl;
  }
  // LOG(INFO) << os.str();
}
/* 
 * if func is succ's residual input, we maintain
 * func's ofl === succ's succ's ifl
 * b/c limited bit shift units on current hardware
 */
 // use union-find
void QNN::CheckQuantizeConstraint() {
  for (auto func : funcs) {
    for (auto pred : func->input_funcs) {
      pred->info->output_fraclen = func->info->input_fraclen;
    }
  }
  for (auto func : funcs) {
    for (auto res_idx : func->info->residual_source) {
      funcs[res_idx - 1]->info->output_fraclen = func->info->output_fraclen;
    }
  }
}

void QNN::VisitExpr_(const FunctionNode* fn) {
  if (fn_top == nullptr) {
    // bookkeep input tensor shape
    const VarNode* op = static_cast<const VarNode*>(fn->params[0].get());
    const auto* rtype = op->checked_type().as<TensorTypeNode>();
    for (auto x : rtype->shape) {
      input_shape.push_back(OpuInfo::Value(x));
    }
    // make sure this if branch not visited again
    fn_top = fn;
  }
  ExprVisitor::VisitExpr_(fn);
}

void QNN::VisitExpr_(const CallNode* call) {
  ExprVisitor::VisitExpr_(call);
}

/*
 * For each constant node, quantize data as weight/bias
 */
void QNN::VisitExpr_(const ConstantNode* op) {
  os << "const in layer[";
  os << fmap_[op]->info->index << "]\n";
  os << "  shape:[";
  for (auto x : op->data.Shape()) {
    os << PrimExpr(static_cast<int>(x)) << " ";
  }
  os << "]\n";
  std::vector<size_t> shape;
  for (auto x : op->data.Shape()) {
    shape.push_back(static_cast<size_t>(x));
  }
  // scalar should be merged, sometime have mutliply(1f, data)
  if (op->is_scalar()) {
    return;
  }
  int64_t size = GetTensorSize(op->data);
  // determine bias/weight
  auto max_iter = std::max_element(shape.begin(), shape.end()); 
  bool is_bias = *max_iter == static_cast<size_t>(size);  
  // bool is_bias = op->data.ToDLPack()->dl_tensor.ndim < 4;
  float* dl = static_cast<float*>(op->data.ToDLPack()->dl_tensor.data);
  // Determine the constant being visited to be weight/bias
  // we use 8-bit weight and 16-bit bias
  // use ndim for convenience for now
  // int ndim = op->data.ToDLPack()->dl_tensor.ndim;
  int wordlen = 8;
  if (is_bias) {
    wordlen = 16;
  } else {
    os << "kernel_layout: " << fmap_[op]->info->kernel_layout << "\n";
  }
  // Search for fraction length that leads to minimal quantization error
  int fl = SearchFraclenUtil(dl, size, wordlen);
  os << "<" << fl << " / " << wordlen << ">\n";
  // write to disk if dump_ is true
  if (dump_) {
    std::string odir = "./dump/";
    struct stat st = {0};
    if (stat(odir.c_str(), &st) == -1) {
      mkdir(odir.c_str(), 0700);
    }
    std::vector<float> tmp(size);
    // if dump quantized weights/bias, quantiz with fl from searching
    if (!dump_unquantized_constant_) {
      Convert(dl, size, wordlen, fl, true, &tmp[0]);
    } else {  // else copy orignal data
      std::memcpy(&tmp[0], dl, size*sizeof(float));
    }
    //save it to file
    std::string pname;
    if (is_bias) {
      pname = "bias";  
      shape = {static_cast<size_t>(size)};
    } else {
      pname = "weights";
      // check weight layout
      tmp = AlterLayout(tmp, shape, fmap_[op]->info->kernel_layout);     
    }
    pname += "_" + std::to_string(fmap_[op]->info->index - 1) + ".npy";
    cnpy::npy_save(odir + pname, &tmp[0], shape);
    // dump unquantized params for checking
    odir = "./dump_raw/";
    struct stat ut = {0};
    if (stat(odir.c_str(), &ut) == -1) {
      mkdir(odir.c_str(), 0700);
    }
    std::memcpy(&tmp[0], dl, size*sizeof(float));
    shape.clear();
    for (auto x : op->data.Shape()) {
      shape.push_back(static_cast<size_t>(x));
    }
    if (!is_bias) {
      tmp = AlterLayout(tmp, shape, fmap_[op]->info->kernel_layout);     
    } else {
      shape = {static_cast<size_t>(size)};
    }
    cnpy::npy_save(odir + pname, &tmp[0], shape);
  }
  // Update weight/bias fraclen in opuinfo(layer)
  OpuInfo* info = fmap_[op]->info;
  info->quantized = true;
  if (is_bias) {
    info->bias_fraclen = fl;
  } else {
    info->weight_fraclen = fl;
  }
  ExprVisitor::VisitExpr_(op);
}

/*
 * transform weight layout to uniform "HWIO"
 */
std::vector<float> QNN::AlterLayout(std::vector<float> src, std::vector<size_t>& shape,
                               std::string layout, std::string target_layout) {
  std::vector<float> ret;
  if (layout == "OIHW") {
    size_t O = shape[0];
    size_t I = shape[1]; 
    size_t H = shape[2];
    size_t W = shape[3];
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        for (size_t i = 0; i < I; i++) {
          for (size_t o = 0; o < O; o++) {
            ret.push_back(src[o*I*H*W+i*H*W+h*W+w]);
          }
        }
      }
    }
    shape = {H,W,I,O};
  } else {
    ret = src;
  }
  return ret;
}

/*
 * Wrappers for searching fraclen that leads to minimal quantization error
 */
int QNN::SearchFraclenUtil(float* dl, size_t size, int wordlen) {
  // Find fraction length for max and min
  auto max_iter = std::max_element(dl, dl+size);
  auto max_index = std::distance(dl, max_iter);
  float max = dl[max_index];
  int fl_max = wordlen - 1 - std::lround(std::log2(std::abs(max)));
  auto min_iter = std::min_element(dl, dl+size);
  auto min_index = std::distance(dl, min_iter);
  float min = dl[min_index];
  int fl_min = wordlen - 1 - std::lround(std::log2(std::abs(min)));
  // Search for best fraction length
  float error = FLT_MAX;
  int fl = 0;
  std::vector<float> errors;
  float* tmp = new float[size];
  // Check the quantization error with fraction length i std::max(fl_max, fl_min)
  for (int i = std::min(FL_SEARCH_MAX - 1, std::min(fl_max, fl_min) - 2);
           i < FL_SEARCH_MAX; i++) {
    float e = Convert(dl, size, wordlen, i, true, tmp);
    if (e < error) {
      error = e;
      fl = i;
    }
    if (errors.size() > 0 && e > errors.back()) {
      break;
    }
    errors.push_back(e);
  }
  delete tmp;
  return fl;
}

int QNN::SearchArrayFraclen(
    const std::vector<tvm::runtime::NDArray>& ndarrays) {
  std::vector<int> fls;
  for (auto ndarray : ndarrays) {
    int64_t size = 1;
    for (auto s : ndarray.Shape()) {
      size *= s;
    }
    float* dl = static_cast<float*>(ndarray.ToDLPack()->dl_tensor.data);
    int fl = SearchFraclenUtil(dl, size, 8);
    fls.push_back(fl);
  }
  auto max_iter = std::min_element(fls.begin(), fls.end());
  return *max_iter;
}

int QNN::SearchFraclen(tvm::runtime::NDArray ndarray) {
  int64_t size = 1;
  for (auto s : ndarray.Shape()) {
    size *= s;
  }
  float* dl = static_cast<float*>(ndarray.ToDLPack()->dl_tensor.data);
  int fl = SearchFraclenUtil(dl, size, 8);
  return fl;
}

/*
 * Quantize data to wordlen-bit fix-point format with fraclen-bit for fraction
 * use_round indicates round/truncation
 * return quantization error corresponding to scheme (wordlen, fraclen, use_round)
 *
 * data_quantized = round(clip_by_max_min(data/step))*step
 */
float QNN::Convert(float* data, size_t size, int wordlen, int fraclen,
    bool use_round, float* data_o) {
  // Precision
  float step = std::pow(2, -fraclen);
  // Bound
  float step_upper_bound = std::pow(2, wordlen-1)-1;
  float step_lower_bound = -std::pow(2, wordlen-1);
  float q_error = 0;
  size_t non_zero_cnt = 0;
  for (size_t i=0; i < size; i++) {
    float n_step = data[i]/step;
    // Round/truncation
    if (use_round) {
      n_step = std::round(n_step);
    } else {
      n_step = std::trunc(n_step);
    }
    // Deal with overflow
    if (n_step > step_upper_bound) {
      n_step = step_upper_bound;
    } else if (n_step < step_lower_bound) {
      n_step = step_lower_bound;
    }
    // Get quantized value
    data_o[i] = n_step*step;
    // Keep record of quantization error
    q_error += (data[i] - data_o[i]) * (data[i] - data_o[i]);
    if (data[i] != data_o[i]) {
      non_zero_cnt += 1;
    }
  }
  return q_error;
}

/*
 * Deploy tvm function to target device
 * Expr body specifies tvm::relay::FunctionNode as function output node
 * input_arrays contain input tensors
 * return output tensor
 */
tvm::runtime::NDArray QNN::Compute(const Expr& body,
    std::vector<tvm::runtime::NDArray> input_ndarrays) {
  IRModule relay_module = IRModule::FromExpr(body);
  relay_module = transform::FuseOps(0)(relay_module);
  //transform::PrintIR(false)(relay_module);
  // Get the updated function.
  auto func = Downcast<Function>(relay_module->Lookup("main"));
  // Deploy with llvm backend on cpu
  Map<tvm::Integer, tvm::Target> targets;
  tvm::Target llvm_tgt = tvm::Target::Create("llvm");
  targets.Set(0, llvm_tgt);
  // Generate code for the updated function.
  std::unique_ptr<backend::GraphCodegen> graph_codegen =
        std::unique_ptr<backend::GraphCodegen>(new backend::GraphCodegen());
  graph_codegen->Init(nullptr, targets);
  graph_codegen->Codegen(func);
  // Get graph JSON and params
  std::string graph_json = graph_codegen->GetJSON();
  std::unordered_map<std::string, tvm::runtime::NDArray> params =
    graph_codegen->GetParams();
  // Build tvm::runtime::Module from input func
  auto lowered_funcs = graph_codegen->GetLoweredFunc();
  tvm::runtime::Module mod = tvm::build(
      lowered_funcs,
      llvm_tgt,
      BuildConfig::Current());
  // Input shape
  /*  
  DLContext ctx = {kDLCPU, 0};
  DLDataType dtype = {kDLFloat, 32, 1};
  */
  const FunctionNode* fn = static_cast<const FunctionNode*>(func.get());
  std::vector<tvm::runtime::NDArray> input_arrays;
  if (fn->params.size() == 0) {
    input_arrays.push_back(input_ndarrays[0]);
  } else {
    for (size_t i = 0; i < fn->params.size(); i++) {
      input_arrays.push_back(input_ndarrays[i]);
    }
  }
  // Get output shape function return type (annotated by InferType pass)
  std::vector<int64_t> oshape = TypeToShape(fn->ret_type);
  auto Out = tvm::runtime::NDArray::Empty(oshape, dtype, ctx);
  // Init GraphRuntime for inference
  tvm::runtime::GraphRuntime* grt = new tvm::runtime::GraphRuntime();
  grt->Init(graph_json, mod, {ctx});
  // Set inputs
  for (size_t i=0; i < input_arrays.size(); i++) {
    grt->SetInput(i, &input_arrays[i].ToDLPack()->dl_tensor);
  }
  // Set params (e.g. conv weights / bias)
  for (auto item : grt->GetInputMap()) {
    //if (item.first == "input") continue;
    if (item.first.find("input") != std::string::npos) continue;
    int in_idx = grt->GetInputIndex(item.first);
    if (params.find(item.first) == params.end()) continue;
    grt->SetInput(in_idx, &params[item.first].ToDLPack()->dl_tensor);
  }
  // Run and collect output
  grt->Run();
  grt->CopyOutputTo(0, &Out.ToDLPack()->dl_tensor);
  delete grt;
  return Out;
}


/*
 * Type -> TensorType -> TensorTypeNode -> std::vector
 */
std::vector<int64_t> QNN::TypeToShape(Type ty) {
  std::vector<int64_t> shape;
  const TensorTypeNode* ttn = static_cast<const TensorTypeNode*>(ty.get());
  for (auto x : ttn->shape) {
    shape.push_back(OpuInfo::Value(x));
  }
  return shape;
}

/*
 * tvm::runtime::NDArray -> data.Shape().reduce(_*_)
 */
int64_t QNN::GetTensorSize(runtime::NDArray data) {
  int64_t size = 1;
  for (auto dim : data.Shape()) {
    size *= dim;
  }
  return size;
}
/*
void QNN::Normalize(const Expr& e) {
  QNorm qnorm;
  qnorm.SetFmap(fmap_);
  qnorm.SetScaleMap(output_dict);
  Expr e_norm = qnorm.Transform(e);
}
*/
}  // namespace relay
}  // namespace tvm
