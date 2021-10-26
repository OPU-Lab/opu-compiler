#include "verifier.h"

namespace tvm{
namespace relay{

bool Verifier::RunTest(Expr& ref) {
  Creator c = Creator();
  c.Prepare();
  
  /*for (auto x : c.outputs) {
    Function func = FunctionNode::make(FreeVars(x), x, Type(), {});
    IRModule relay_module = IRModule::FromExpr(func);
    relay_module = transform::InferType()(relay_module);
    func = Downcast<Function>(relay_module->Lookup("main"));
    auto f = reinterpret_cast<const FunctionNode*>(func.get());
  std::cout << f->ret_type << "\n";
  }
  auto f = reinterpret_cast<const FunctionNode*>(ref.get());
  std::cout << f->ret_type << "\n";
  exit(1);*/
  
  // get reference result
  Deployer d0 = Deployer();
  std::string layout_ref = d0.GetLayout(ref);
  std::vector<int64_t> ishape = d0.GetInputShape(ref);
  int64_t N, C, H, W;
  if (layout_ref == "NCHW") {
    N = ishape[0];
    C = ishape[1];
    H = ishape[2];
    W = ishape[3];
  } else {
    N = ishape[0];
    H = ishape[1];
    W = ishape[2];
    C = ishape[3];
  }

  std::vector<tvm::runtime::NDArray> inputs = d0.GetInputs(ishape);
  std::vector<Expr> out0 = d0.GetOutputs(ref);
  for (auto e : out0) {
    d0.Run(e, inputs, true);
  }
  
  // get ir result
  Deployer d = Deployer();
  std::vector<int64_t> target_shape;
  if (c.target_layout == "NHWC") {
    target_shape = {N, H, W, C};
  } else {
    target_shape = {N, C, H, W};
  }
  
  /*tvm::runtime::NDArray A =
    tvm::runtime::NDArray::Empty(target_shape, d.dtype, d.ctx);
  cnpy::NpyArray arr = cnpy::npy_load("input.npy");
  std::vector<float> ret = arr.as_vec<float>();
  std::vector<float> ret;
  for (int64_t n = 0; n < N; n++) {
    for (int64_t h = 0; h < H; h++) {
      for (int64_t w = 0; w < W; w++) {
        for (int64_t c = 0; c < C; c++) {
          ret.push_back(loaded_data[n*C*H*W+c*H*W+h*W+w]);
        }
      }
    }
  }
  float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
  std::memcpy(data, &ret[0], sizeof(float)*d.GetTensorSize(A));
  std::vector<tvm::runtime::NDArray> t = {A};*/
  
  /*
  for (int i=0;i<9;i++){
    std::cout << "^^^^^^^^^^^^^^^^^" << i << "\n";
    auto z = c.emap_[i];
    Function func = FunctionNode::make(FreeVars(z), z, Type(), {});  
    d.Run(func, inputs);
  }
  exit(1);
  */
  for (auto z : c.outputs) {
    Function func = FunctionNode::make(FreeVars(z), z, Type(), {});  
    d.Run(func, inputs);
  } 
  
  // compare
  auto pairs = FindMatch(d0.results, d.results);
  for (auto pair : pairs) {
    std::cout << "ref.results[" << pair.first << "] v.s. "
              << "ir.results[" << pair.second << "]\n";
    d.IsEqual(d0.results[pair.first], d.results[pair.second]);
  }
  
  return true;
}


std::vector<tvm::runtime::NDArray> Verifier::Deployer::GetInputs(std::vector<int64_t> input_shape) {
  tvm::runtime::NDArray A =
    tvm::runtime::NDArray::Empty(input_shape, dtype, ctx);
  float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
  // load from npy
  /*
  cnpy::NpyArray arr = cnpy::npy_load("input.npy");
  std::vector<float> loaded_data = arr.as_vec<float>();
  std::memcpy(data, &loaded_data[0], sizeof(float)*GetTensorSize(A));
  */
  // generate random
  int64_t size = GetTensorSize(A);
  std::vector<float> data_rand;
  srand(time(0));
  for (int i = 0; i < size; i++)
    data_rand.push_back(static_cast<float>(rand()%100)/100);
  std::memcpy(data, &data_rand[0], sizeof(float)*size);

  std::vector<tvm::runtime::NDArray> ret;
  ret = {A};
  return ret;
}

/*
 * find potentially equivalent ndarray pairs
 * in terms of shape (criterion need to be enhanced)
 */
std::vector<std::pair<size_t, size_t>> Verifier::FindMatch(
    std::vector<tvm::runtime::NDArray> vec_a,
    std::vector<tvm::runtime::NDArray> vec_b) {
  std::vector<std::pair<size_t, size_t>> pairs;
  
  CHECK_EQ(vec_a.size(), vec_b.size()) << ": unmatched output counts\n";
  
  if (vec_a.size() == 1) {
    return {{0, 0}};
  }
  for (size_t i = 0; i < vec_a.size(); i++) {
    for (size_t j =0; j < vec_b.size(); j++) {
      if (EqShape(vec_a[i], vec_b[j])) {
        pairs.push_back({i, j});
        break;
      }
    }
  }
  return pairs;
}

bool Verifier::EqShape(tvm::runtime::NDArray A, tvm::runtime::NDArray B) {
  bool eq = true;
  std::vector<int64_t> shape_a = A.Shape();
  std::vector<int64_t> shape_b = B.Shape();
  for (size_t i = 0; i < shape_a.size(); i++) {
    if (shape_a[i] != shape_b[i]) {
      eq = false;
      break;
    }
  }
  return eq;
}

/*
 * check reference function ret_type
 * decouple to multiple outputs if ret_type == TupleNode
 */
std::vector<Expr> Verifier::Deployer::GetOutputs(Expr& body) {
  std::vector<Expr> outputs;
  if (body.get()->IsInstance<CallNode>()) {
    // if expr is not a function, wrap to function
    // reconstruct from IR is such case
    Function func = FunctionNode::make(FreeVars(body), body, Type(), {});
    outputs.push_back(func);  
  } else if (body.get()->IsInstance<FunctionNode>()) {
    // if expr is function, check return type
    const FunctionNode* fn = reinterpret_cast<const FunctionNode*>(body.get());
    if (fn->body.get()->IsInstance<TupleNode>()) {
      const TupleNode* tuple = reinterpret_cast<const TupleNode*>(fn->body.get());
      for (auto arg : tuple->fields) {
        Function func = FunctionNode::make(FreeVars(arg), arg, Type(), {});
        outputs.push_back(func);  
      }
    } else {
      outputs.push_back(body);  
    }
  }
  return outputs;
}
 
bool Verifier::Deployer::IsEqual(tvm::runtime::NDArray A, tvm::runtime::NDArray B) {
  auto a_iter = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
  auto b_iter = static_cast<float*>(B.ToDLPack()->dl_tensor.data);
  int64_t size_a = GetTensorSize(A);
  int64_t size_b = GetTensorSize(B);
  bool eq = true;
  if (size_a != size_b) {
    std::cout << "Unmatched output size!\n";
    eq = false;
  } else {
    std::vector<float> dif;
    for (int64_t i = 0; i < size_a; i++) {
      dif.push_back(std::abs(a_iter[i] - b_iter[i]));
    }
    auto max_iter = std::max_element(dif.begin(), dif.end());
    std::cout << "[CHECK] Max abs dif = " << *max_iter << "\n";
    float dif_avg = std::accumulate(dif.begin(), dif.end(), 0.0) / size_a;
    std::cout << "[CHECK] Avg abs dif = " << dif_avg << "\n\n";
  }
  return eq;
}

std::vector<int64_t> Verifier::Deployer::GetInputShape(Expr& body) {
  std::vector<int64_t> shape;
  const FunctionNode* fn = reinterpret_cast<const FunctionNode*>(body.get());
  const VarNode* op = static_cast<const VarNode*>(fn->params[0].get());
  const auto* rtype = op->checked_type().as<TensorTypeNode>();
  for (auto x : rtype->shape) {
    shape.push_back(OpuInfo::Value(x));
  }
  return shape;
}

/*
 * infer global layout with the first met conv2d's data layout
 * (assume no data layout alteration within graph globally)
 */
std::string Verifier::Deployer::GetLayout(Expr& body) {
  struct LayoutVisitor : ExprVisitor {
    std::string layout = "";
    void VisitExpr_(const CallNode* call) final {
      if (layout != "") return;
      else if (call->op == Op::Get("nn.conv2d")) {
        auto attrs = call->attrs.as<Conv2DAttrs>();
        layout = attrs->data_layout;
      } else {
        ExprVisitor::VisitExpr_(call);
      }
    }
  } visitor;
  visitor(body);
  return visitor.layout;
}

int64_t Verifier::Deployer::GetTensorSize(runtime::NDArray data) {
  int64_t size = 1;
  for (auto dim : data.Shape()) {
    size *= dim;
  }
  return size;
}

/*
 * run a function
 */
void Verifier::Deployer::Run(Expr& body,
    std::vector<tvm::runtime::NDArray> input_ndarrays,
    bool raw) {
  IRModule relay_module = IRModule::FromExpr(body);
  transform::PrintIR(false)(relay_module);
  if (raw) {
    relay_module = transform::SimplifyInference()(relay_module);
    /*relay_module = transform::FoldConstant()(relay_module);
    relay_module = transform::FoldScaleAxis()(relay_module);
    relay_module = transform::CanonicalizeCast()(relay_module);
    relay_module = transform::CanonicalizeOps()(relay_module);*/
  }
  relay_module = transform::FuseOps(0)(relay_module);
  // 
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
  std::vector<int64_t> oshape;
  const TensorTypeNode* ttn = static_cast<const TensorTypeNode*>(fn->ret_type.get());
  for (auto x : ttn->shape) {
    oshape.push_back(OpuInfo::Value(x));
  }
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
    if (item.first == "input") continue;
    int in_idx = grt->GetInputIndex(item.first);
    grt->SetInput(in_idx, &params[item.first].ToDLPack()->dl_tensor);
  }
  // Run and collect output
  grt->Run();
  grt->CopyOutputTo(0, &Out.ToDLPack()->dl_tensor);
  delete grt;
  results.push_back(Out);  
}

void Verifier::Creator::Prepare() {
  std::ifstream inj("OPU_IR.json");
  std::string line;
  while (std::getline(inj, line)) {
    std::istringstream iss(line);
    json jt;
    iss >> jt;
    OpuInfo* info = new OpuInfo();
    info->from_json(jt);
    // if (info->index > 52) break;
    BuildLayer(info);
  }
  inj.close();
  // exit(1);
}

void Verifier::Creator::Update(int index, Expr expr) {
  emap_[index] = expr;
}

std::vector<IndexExpr> Verifier::Creator::GetIndexExprVec(std::vector<int64_t> ishape) {
  std::vector<IndexExpr> shape;
  for (auto x : ishape) {
    shape.push_back(IndexExpr(static_cast<int>(x)));
  }
  return shape;  
}

void Verifier::Creator::BuildLayer(OpuInfo* info) {
  info_map_[info->index] = info;
  // get input
  Expr data;
  if (info->input_layer.size() == 1) {
    if (info->input_layer[0] == 0) {
      if (info->extra_pre_padding == 1) {
        info->input_size[1] -= (info->padding_size[0] + info->padding_size[1]);
        info->input_size[2] -= (info->padding_size[2] + info->padding_size[3]);
      }
      std::vector<int64_t> input_shape = info->input_size;
      if (target_layout == "NCHW") {
        input_shape = {info->input_size[0], info->input_size[3],
                       info->input_size[1], info->input_size[2]};
      }
      auto var_type = TensorTypeNode::make(
          GetIndexExprVec(input_shape), DataType::Float(32));
      Var var = VarNode::make("input", var_type);
      if (info->post_padding_mode == 1 && info->extra_pre_padding == 1) {
        data = MakePad(var, info->padding_size);
        Update(0, data);
      } else {
        Update(0, var);
      }
    }
    data = emap_[info->input_layer[0]];
  } else {  // concat
    Array<Expr> inputs;
    for (auto idx : info->input_layer) {
      inputs.push_back(emap_[idx]);
    }
    auto tuple = TupleNode::make(inputs);
    // concat inputs "NHWC" via axis = 3 "C"
    data = MakeConcatenate(tuple, concat_axis);
  }
  // pre-padding
  if (info->post_padding_mode == 1) {
    // if post padding, then pre remove padding is needed here
    int sum = 0;
    for (auto x : info->pre_remove_padding_size)
      sum += x;
    if (info->extra_pre_padding == 0 && sum > 0)
      data = MakeDePad(data, info->pre_remove_padding_size);    
  } else if (!IsEmpty(info->padding_size)) {
    data = MakePad(data, info->padding_size);
  }  
  // core
  if (info->type == 1) {  // conv2d
    std::vector<int64_t> wshape;
    if (target_layout == "NCHW") {  // OIHW
      wshape = {info->output_size[3], info->input_size[3], info->ker_size[1], info->ker_size[2]};
    } else {  // NHWC -> weight shape HWIO
      wshape = {info->ker_size[1], info->ker_size[2], info->input_size[3], info->output_size[3]};
    }
    if (info->group > 1) {  // depthwise-conv2d
      if (target_layout == "NCHW") wshape[1] = 1;
      else wshape[2] = 1;
    }
    Constant weight = LoadConstant(info->index, "weights", wshape);
    // stride
    Array<IndexExpr> strides = {
        ToIndexExpr(info->ker_stride[1]), ToIndexExpr(info->ker_stride[2])};
    // zero padding, we have extra pre-padding/post-padding
    auto padding = {relay::IndexExpr(0), relay::IndexExpr(0),
                       relay::IndexExpr(0), relay::IndexExpr(0)};
    auto dilation = {relay::IndexExpr(1), relay::IndexExpr(1)};
    int groups = info->group;
    auto channels = ToIndexExpr(info->output_size[3]);
    auto kernel_size = {ToIndexExpr(info->ker_size[1]), ToIndexExpr(info->ker_size[2])};
    std::string data_layout = target_layout;
    std::string kernel_layout = (target_layout == "NCHW")? "OIHW" : "HWIO";
    std::string out_layout = data_layout;
    auto out_dtype = runtime::DataType::Float(32);
    data = Conv2D(data, weight, strides, padding, dilation, groups,
                         channels, kernel_size, data_layout, kernel_layout,
                         out_layout, out_dtype);
    // bias
    Constant bias = LoadConstant(info->index, "bias", {info->output_size[3]});
    // target dimension = 4 "NHWC", match axis = 3 "C"
    data = Add(data, ExpandBiasToMatchAxis(bias, 4, {concat_axis}));
  } else if (info->type == 0) {  // dense
    // weight shape Cout, Cin
    std::vector<int64_t> wshape = {info->output_size[3], info->input_size[3]};
    Constant weight = LoadConstant(info->index, "weights", wshape);
    IndexExpr units = IndexExpr(static_cast<int>(info->output_size[3]));
    auto out_dtype = runtime::DataType::Float(32);
    // flatten first -> N, Cin
    data = CallNode::make(Op::Get("nn.batch_flatten"), {data}, Attrs(), {});
    data = Dense(data, weight, units, out_dtype);
    Constant bias = LoadConstant(info->index, "bias", {info->output_size[3]});
    data = Add(data, ExpandBiasToMatchAxis(bias, 2, {1}));
  } else {
      
  }
  // post-processing
  // TODO: pool-relu =/= relu-pool, especially for avgpool!
  if (info->res_position == 0) {  // no residual
    data = MakeResidualAdd(data, info->residual, info->residual_source);
    data = MakeActivation(data, info->activation_type);
    data = MakePooling(data, info->pooling_type, info->pooling_size,
        info->pooling_stride, info->pool_padding_size);
  } else if (info->res_position == 1) {  // relu - res - pool(option)
    data = MakeActivation(data, info->activation_type);
    data = MakeResidualAdd(data, info->residual, info->residual_source);
    data = MakePooling(data, info->pooling_type, info->pooling_size,
        info->pooling_stride, info->pool_padding_size);
  } else if (info->res_position == 2) {  // pool - res - relu(option)
    data = MakePooling(data, info->pooling_type, info->pooling_size,
        info->pooling_stride, info->pool_padding_size);
    data = MakeResidualAdd(data, info->residual, info->residual_source);
    data = MakeActivation(data, info->activation_type);    
  } else {  // pool - relu - res
    data = MakeActivation(data, info->activation_type);
    data = MakePooling(data, info->pooling_type, info->pooling_size,
        info->pooling_stride, info->pool_padding_size);
    data = MakeResidualAdd(data, info->residual, info->residual_source);   
  }
  // upsampling
  if (info->upsample != 0) {
    data = MakeUpsampling(data, info->upsample, info->upsample_method);
  }
  // post-padding
  if (info->post_padding_mode == 1 && info->post_padding == 1) {
    data = MakePad(data, info->post_padding_size);
  }
  // check quantization
  if (info->residual) {
    CHECK_EQ(
        info_map_[info->residual_source[0]]->output_fraclen,
        info->output_fraclen) << " layer[" << info->index
        << "]: residual input layer[" << info->residual_source[0]
        << "]'s output_fraclen v.s. self's output fraclen\n";
  }
  // bookkeep layer output expr
  Update(info->index, data);
  // bookkeep global outputs  
  if (info->output_layer.size() == 1 && info->output_layer[0] == -1) {
    outputs.push_back(data);
  }
  
  // debug print
  /*
  std::ostringstream os;
  info->dump(os);
  std::cout << os.str();
  IRModule relay_module = IRModule::FromExpr(data);
  transform::PrintIR(false)(relay_module);
  */
}

Expr Verifier::Creator::MakeUpsampling(Expr data, int scale, std::string method) {
  auto attrs = make_object<UpSamplingAttrs>();
  attrs->scale_h = scale;
  attrs->scale_w = scale;
  attrs->layout = target_layout;
  attrs->method = method;
  attrs->align_corners = true;
  return CallNode::make(Op::Get("nn.upsampling"), {data}, Attrs(attrs), {});
}

Expr Verifier::Creator::MakeResidualAdd(Expr data, int has_residual, std::vector<int64_t> residual_source) {
  Expr ret = data;
  if (has_residual == 1) {
    int res_src_index = residual_source[0];
    Expr res_src_expr = emap_[res_src_index];
    ret = Add(data, res_src_expr);
  }
  return ret;  
}

Expr Verifier::Creator::MakePooling(Expr data, int type, std::vector<int64_t> pooling_size,
    std::vector<int64_t> pooling_stride, std::vector<int64_t> pool_padding_size) {
  Array<IndexExpr> pool_size = {ToIndexExpr(pooling_size[1]), ToIndexExpr(pooling_size[2])};
  Array<IndexExpr> strides = {ToIndexExpr(pooling_stride[1]), ToIndexExpr(pooling_stride[2])};
  // top, left, bottom, right
  Array<IndexExpr> padding = {ToIndexExpr(pool_padding_size[0]), ToIndexExpr(pool_padding_size[1]),
    ToIndexExpr(pool_padding_size[2]), ToIndexExpr(pool_padding_size[3])};
  std::string layout = target_layout;
  bool ceil_mode = false;
  Expr ret = data;
  if (type == 1) {  // max
    ret = MaxPool2D(ret, pool_size, strides, padding, layout, ceil_mode);
  } else if (type == 2) {  // avg
    bool count_include_pad = false;
    ret = AvgPool2D(ret, pool_size, strides, padding, layout, ceil_mode, count_include_pad);  
  }
  return ret;
}

Expr Verifier::Creator::MakeActivation(Expr data, int type) {
  Expr ret = data;
  if (type == 1) {  // relu 
    ret = CallNode::make(Op::Get("nn.relu"), {ret}, Attrs(), {});
  } else if (type == 2) {  // prelu, set alpha=0.1
    auto attrs = make_object<LeakyReluAttrs>();
    attrs->alpha = 0.1;
    ret = CallNode::make(Op::Get("nn.leaky_relu"), {ret}, Attrs(attrs), {});
  } else if (type == 6) {
    auto attrs = make_object<ClipAttrs>();
    attrs->a_min = 0.0;
    attrs->a_max = 6.0;
    ret = CallNode::make(Op::Get("clip"), {ret}, Attrs(attrs), {});
  }
  return ret;
}

Constant Verifier::Creator::LoadConstant(int index, std::string name, std::vector<int64_t> shape) {
  std::string filename = "dump/";
  filename += name + "_" + std::to_string(index - 1) + ".npy";
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  std::vector<float> loaded_data = arr.as_vec<float>();
  tvm::runtime::NDArray A =
    tvm::runtime::NDArray::Empty(shape, dtype, ctx);
  float* data = static_cast<float*>(A.ToDLPack()->dl_tensor.data);
  int64_t size = 1;
  for (auto dim : A.Shape()) {
    size *= dim;
  }
  if (size != static_cast<int64_t>(loaded_data.size())) {
    std::cout << filename << "\n";
    std::cout << "get :" << loaded_data.size() << " v.s. target: " << size << "\n";
    exit(1);
  }
  // tvm NCHW can only convolve with OIHW layout ?
  // dumped weight is HWIO in default
  std::vector<float> ret;
  if (name == "weights" && shape.size() == 4 && target_layout == "NCHW") {
    size_t O = shape[0];
    size_t I = shape[1]; 
    size_t H = shape[2];
    size_t W = shape[3];
    for (size_t o = 0; o < O; o++) {
      for (size_t i = 0; i < I; i++) {
        for (size_t h = 0; h < H; h++) {
          for (size_t w = 0; w < W; w++) {          
            ret.push_back(loaded_data[h*W*I*O+w*I*O+i*O+o]);
          }
        }
      }
    }
  } else {
    ret = loaded_data;
  }
  std::memcpy(data, &ret[0], sizeof(float)*size);
  return ConstantNode::make(A);  
}

bool Verifier::Creator::IsEmpty(std::vector<int64_t> vec) {
  int64_t sum = 0;
  for (auto x : vec) {
    sum += x;
  }
  return vec.size() == 0 || sum == 0;
}

Expr Verifier::Creator::MakePad(Expr data, std::vector<int64_t> padding_size) {
  double pad_value = 0.0;
  std::string pad_mode = "constant";
  // OPU IR assume data layout NHWC
  Array<Array<IndexExpr>> pad_width;
  if (target_layout == "NHWC") {
    pad_width = {
      {IndexExpr(0), IndexExpr(0)},
      {ToIndexExpr(padding_size[0]), ToIndexExpr(padding_size[1])},
      {ToIndexExpr(padding_size[2]), ToIndexExpr(padding_size[3])},
      {IndexExpr(0), IndexExpr(0)}
    };
  } else {
    pad_width = {
      {IndexExpr(0), IndexExpr(0)},
      {IndexExpr(0), IndexExpr(0)},
      {ToIndexExpr(padding_size[0]), ToIndexExpr(padding_size[1])},
      {ToIndexExpr(padding_size[2]), ToIndexExpr(padding_size[3])}
    };  
  }
  return Pad(data, pad_width, pad_value, pad_mode);
}

Expr Verifier::Creator::MakeDePad(Expr data, std::vector<int64_t> padding_size) {
  IRModule relay_module = IRModule::FromExpr(data);
  relay_module = transform::InferType()(relay_module);
  auto func = Downcast<Function>(relay_module->Lookup("main"));
  const FunctionNode* fn = static_cast<const FunctionNode*>(func.get());
  const auto* rtype = fn->ret_type.as<TensorTypeNode>();
  std::vector<int64_t> fm_shape;
  for (auto x : rtype->shape) {
    fm_shape.push_back(OpuInfo::Value(x));
  }
  Array<Integer> begin;
  Array<Integer> end;
  Array<Integer> strides;
  if (target_layout == "NHWC") {
    begin = {
        0, 
        static_cast<int>(padding_size[0]), 
        static_cast<int>(padding_size[2]), 
        0
        };
    end = {
        static_cast<int>(fm_shape[0]), 
        static_cast<int>(fm_shape[1] - padding_size[1]), 
        static_cast<int>(fm_shape[2] - padding_size[3]), 
        static_cast<int>(fm_shape[3])
        };
  } else {
    begin = {
        0,
        0,
        static_cast<int>(padding_size[0]), 
        static_cast<int>(padding_size[2])
        };
    end = {
        static_cast<int>(fm_shape[0]),
        static_cast<int>(fm_shape[1]),
        static_cast<int>(fm_shape[2] - padding_size[1]), 
        static_cast<int>(fm_shape[3] - padding_size[3])
        };
  }
  strides = {1,1,1,1};
  return MakeStridedSlice(data, begin, end, strides);
}

}  // namespace relay    
}  // namespace tvm