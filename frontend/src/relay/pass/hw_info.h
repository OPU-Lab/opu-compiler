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
#ifndef SRC_RELAY_PASS_HW_INFO_H_
#define SRC_RELAY_PASS_HW_INFO_H_

#include <tvm/expr_operator.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/ir/attrs.h>

#include <fstream>
#include <string>
#include <vector>

#include "./pattern_util.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
typedef int64_t opu_int;
namespace tvm {
namespace relay {
class OpuInfo {
 public:
  // data structures for backend
  opu_int type;
  std::vector<opu_int> input_layer;
  std::vector<opu_int> output_layer;
  std::vector<opu_int> input_size;
  std::vector<opu_int> output_size;
  std::vector<opu_int> ker_size;
  std::vector<opu_int> ker_stride;
  std::vector<opu_int> pooling_size;
  std::vector<opu_int> pooling_stride;
  opu_int pooling_type {0};
  std::vector<opu_int> pool_padding_size;
  std::vector<opu_int> padding_size;
  opu_int activation_type {0};
  opu_int residual {0};
  std::vector<opu_int> residual_source;
  std::vector<opu_int> output_choice {{3, -1}};
  opu_int res_position {3};
  opu_int group {0};
  opu_int channel_shuffle {0};
  opu_int upsample {0};
  // post padding
  opu_int extra_pre_padding {0};  // for testing purpose only
  std::vector<opu_int> post_padding_size;
  std::vector<opu_int> pre_remove_padding_size;
  opu_int post_padding {0};
  opu_int post_padding_mode {0};
  // temporary data structures
  std::string data_layout {""};
  std::string kernel_layout {""};
  std::vector<opu_int> dilation;
  std::string upsample_method {""};
  opu_int index;
  std::vector<std::string> op_dfs_order;
  std::unordered_map<std::string, int> op_dfs_order_idx_map;
  std::vector<opu_int> explicit_padding_size;
  opu_int explicit_pad {0};
  // quantization
  bool quantized {false};
  opu_int input_fraclen;
  opu_int output_fraclen;
  opu_int weight_fraclen;
  opu_int bias_fraclen;

  static opu_int Value(const PrimExpr& e) {
    return static_cast<opu_int>(static_cast<const IntImmNode*>(e.get())->value);
  }

  // bookkeep operator in dfs order
  void BookKeepOpOrder(std::string op_name) {
    op_dfs_order_idx_map[op_name] = op_dfs_order.size();  
    op_dfs_order.push_back(op_name);
  }
  
  // locally do sanity check and param regularization
  void Canonicalize() {
    //  input_layer / output_layer
    if (input_layer.size() == 0) {
      input_layer.push_back(0);
    } else {
      // 0-index to 1-index, need to uniform
      /*for(auto& u: input_layer){
        u++;
      }*/
    }
    /*for(auto& u: output_layer){
      u++;
    }*/
    if (output_layer.size() == 0) {
      output_layer.push_back(-1);
    }
    //  input_size / output_size / ker_size
    if (data_layout == "NCHW" || data_layout == "") {
      // -> NHWC
      opu_int C_i = input_size[1];
      input_size.erase(input_size.begin()+1);
      input_size.push_back(C_i);
      opu_int C_o = output_size[1];
      output_size.erase(output_size.begin()+1);
      output_size.push_back(C_o);
    }
    if (input_size.size() == 2) {
      input_size.insert(input_size.begin()+1, 1);
      input_size.insert(input_size.begin()+1, 1);
    }
    if (output_size.size() == 2) {
      output_size.insert(output_size.begin()+1, 1);
      output_size.insert(output_size.begin()+1, 1);
    }
    if (kernel_layout == "OIHW") {
      // pytorch: OIHW -> DHW
      ker_size.erase(ker_size.begin(), ker_size.begin()+2);
      ker_size.insert(ker_size.begin(), 1);
    }
    // ker_size
    if (ker_size.size() == 0) {
      ker_size = {1, 1, 1};
    }
    for (size_t i = 0; i < 3 - ker_size.size(); i++) {
      ker_size.insert(ker_size.begin(), 1);
    }
    // padding_size
    if (explicit_pad == 1) {
      padding_size = explicit_padding_size;
    }
    if (padding_size.size() == 0) {
      padding_size = std::vector<opu_int>(4, 0);
    } else if (padding_size.size() < 4) {
      size_t sz = padding_size.size();
      for (size_t i = 0; i < 4 - sz; i++) {
        padding_size.insert(padding_size.begin(), 0);
      }
      CHECK_EQ(padding_size.size(), 4);
    }
    // ker_stride
    if (ker_stride.size() == 0) {
      ker_stride = {1, 1, 1};
    }
    for (size_t i = 0; i < 3 - ker_stride.size(); i++) {
      ker_stride.insert(ker_stride.begin(), 1);
    }
    // pool
    if (pooling_type == 0) {
      // -> KdKhKw
      pooling_size = std::vector<opu_int>(3, 1);
      pooling_stride = std::vector<opu_int>(3, 1);
      pool_padding_size = std::vector<opu_int>(4, 0);
    } else if (pooling_size.size() == 2) {
      pooling_size.insert(pooling_size.begin(), 1);
      pooling_stride.insert(pooling_stride.begin(), 1);
    }
    if (pool_padding_size.size() == 0) {
      pool_padding_size = std::vector<opu_int>(4, 0);
    } else if (pool_padding_size.size() < 4) {
      for (size_t j = pool_padding_size.size(); j < 4; j++) {
        pool_padding_size.push_back(0);
      }
    }
    // residual
    if (residual_source.size() != 0) {
      residual = 1;
      // encode res_position from operator names in dfs order
      int res_idx = op_dfs_order_idx_map["residual_add"];
      int act_idx = res_idx;
      int pool_idx = res_idx;
      for (auto item : op_dfs_order_idx_map) {
        if (item.first.find("relu") != std::string::npos) {
          act_idx = item.second;
        } else if (item.first.find("pool") != std::string::npos) {
          pool_idx = item.second;
        }
      }
      // == indicates that act/pool does not exist
      /*if (res_idx < act_idx && res_idx < pool_idx) {  // after all
        res_position = 3;  
      } else if (res_idx < act_idx) {  // relu - res - pool(option)
        res_position = 1;
      } else if (res_idx < pool_idx) {  // pool - res - relu(option)
        res_position = 2;
      }*/
      /*if (res_idx > act_idx && res_idx > pool_idx) {  // after all
        res_position = 0;  
      } else if (res_idx > act_idx) {  // relu - res - pool
        res_position = 1;
      } else if (res_idx <= act_idx && res_idx <= pool_idx) {  // res - act - pool
        res_position = 2;
      }*/
      if (res_idx <= act_idx && res_idx <= pool_idx) {  // after all
        res_position = 2;  
      } else if (res_idx <= act_idx) {  // relu - res - pool
        res_position = 1;
      } else if (res_idx >= act_idx && res_idx >= pool_idx) {  // res - act - pool
        res_position = 0;
      }
      std::cout << "<<<< res_idx:" << res_idx
         << " act_idx:" << act_idx
         << " pool_idx:" << pool_idx
         << " >>>>> " << res_position << "\n";
      // output choice for residual
      //output_choice[1] = 3;
    }
  }

  void dump(std::ostringstream& os) {
    os << "======group" << index << "======\n";
    os << "type:" << type << "\n";
    os << "input_layer:[";
    for (auto u : input_layer) {
      os << u <<",";
    }
    os << "]\n";
    os << "output_layer:[";
    for (auto u : output_layer) {
      os << u <<",";
    }
    os << "]\n";
    os << "data_layout:" << data_layout << "\n";
    os << "input_size:[";
    for (auto u : input_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "output_size:[";
    for (auto u : output_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "ker_size:[";
    for (auto u : ker_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "ker_stride:[";
    for (auto u : ker_stride) {
      os << u <<",";
    }
    os << "]\n";
    os << "pooling_size:[";
    for (auto u : pooling_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "pooling_stride:[";
    for (auto u : pooling_stride) {
      os << u <<",";
    }
    os << "]\n";
    os << "pooling_type:" << pooling_type << "\n";
    os << "pool_padding_size:[";
    for (auto u : pool_padding_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "padding_size:[";
    for (auto u : padding_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "post_padding_size:[";
    for (auto u : post_padding_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "pre_remove_padding_size:[";
    for (auto u : pre_remove_padding_size) {
      os << u <<",";
    }
    os << "]\n";
    os << "activation_type:" << activation_type << "\n";
    os << "residual_source:[";
    for (auto u : residual_source) {
      os << u <<",";
    }
    os << "]\n";
    os << "output_choice:[";
    for (auto u : output_choice) {
      os << u <<",";
    }
    os << "]\n";
    os << "res_position:" << res_position << "\n";
    os << "residual:" << residual << "\n";
    os << "group:" << group << "\n";
    os << "channel_shuffle:" << channel_shuffle << "\n";
    os << "upsample:" << upsample << "\n";
    // tmp
    os << "post_padding:" << post_padding << "\n";
    os << "extra_pre_padding:" << extra_pre_padding << "\n";
    os << "kernel_layout:" << kernel_layout << "\n";
    os << "dilation:[";
    for (auto u : dilation) {
      os << u <<",";
    }
    os << "]\n";
    // quantization
    if (quantized) {
      os << "input_fraclen:" << input_fraclen << "\n";
      os << "output_fraclen:" << output_fraclen << "\n";
      os << "weight_fraclen:" << weight_fraclen << "\n";
      os << "bias_fraclen:" << bias_fraclen << "\n";
    }
  }
  
  void to_json(json& j) {
    j["index"] = index;
    j["type"] = type;
    j["input_layer"] = input_layer;    
    j["output_layer"] = output_layer;    
    j["input_size"] = input_size;    
    j["output_size"] = output_size;    
    j["ker_size"] = ker_size;    
    j["ker_stride"] = ker_stride;    
    j["pooling_size"] = pooling_size;    
    j["pooling_stride"] = pooling_stride;    
    j["pooling_type"] = pooling_type; 
    j["pool_padding_size"] = pool_padding_size; 
    j["padding_size"] = padding_size; 
    j["activation_type"] = activation_type; 
    j["residual"] = residual; 
    j["residual_source"] = residual_source; 
    j["output_choice"] = output_choice; 
    j["res_position"] = res_position;    
    j["group"] = group;    
    j["channel_shuffle"] = channel_shuffle;    
    j["upsample"] = upsample;
    j["data_layout"] = data_layout;
    j["kernel_layout"] = kernel_layout;
    j["dilation"] = dilation;
    j["upsample_method"] = upsample_method;
    j["input_fraclen"] = input_fraclen;
    j["output_fraclen"] = output_fraclen;
    j["weight_fraclen"] = weight_fraclen;
    j["bias_fraclen"] = bias_fraclen;
    
    j["post_padding_size"] = post_padding_size; 
    j["pre_remove_padding_size"] = pre_remove_padding_size; 
    j["post_padding"] = post_padding;
    j["post_padding_mode"] = post_padding_mode;
    j["extra_pre_padding"] = extra_pre_padding;
  }
  
  void from_json(const json& j) {
    j.at("index").get_to(index);
    j.at("type").get_to(type);
    j.at("input_layer").get_to(input_layer);
    j.at("output_layer").get_to(output_layer);
    j.at("input_size").get_to(input_size);
    j.at("output_size").get_to(output_size);
    j.at("ker_size").get_to(ker_size);
    j.at("ker_stride").get_to(ker_stride);
    j.at("pooling_size").get_to(pooling_size);
    j.at("pooling_stride").get_to(pooling_stride);
    j.at("pooling_type").get_to(pooling_type);
    j.at("pool_padding_size").get_to(pool_padding_size);
    j.at("padding_size").get_to(padding_size);
    j.at("activation_type").get_to(activation_type);
    j.at("residual").get_to(residual);
    j.at("residual_source").get_to(residual_source);
    j.at("output_choice").get_to(output_choice);
    j.at("res_position").get_to(res_position);
    j.at("group").get_to(group);
    j.at("channel_shuffle").get_to(channel_shuffle);
    j.at("upsample").get_to(upsample);
    j.at("data_layout").get_to(data_layout);
    j.at("kernel_layout").get_to(kernel_layout);
    j.at("dilation").get_to(dilation);
    j.at("upsample_method").get_to(upsample_method);
    j.at("input_fraclen").get_to(input_fraclen);
    j.at("output_fraclen").get_to(output_fraclen);
    j.at("weight_fraclen").get_to(weight_fraclen);
    j.at("bias_fraclen").get_to(bias_fraclen);
    
    j.at("post_padding_size").get_to(post_padding_size);
    j.at("pre_remove_padding_size").get_to(pre_remove_padding_size);
    j.at("post_padding").get_to(post_padding);
    j.at("post_padding_mode").get_to(post_padding_mode);
    j.at("extra_pre_padding").get_to(extra_pre_padding);
  }
};

class OpuInfoCollection {
 public:
  std::vector<OpuInfo*> collection;
  bool quantized {true};
  bool post_padding {false};

  void Add(OpuInfo* info) {
    collection.push_back(info);
    quantized &= info->quantized;
  }
  
  void GlobalCanonicalize() {
    CanonicalizeResidual();
    LegalizeDensePredOutputShape();
  }
  
  void LegalizeDensePredOutputShape() {
    for (auto info : collection) {
      if (info->type == 0) {
        // nn.batch_flatten with possibel proceeding trasnpose
        // before nn.dense are currently mapped to its precessor
        // layer. So we change the output size to be that before
        // flatten to fit backend.
        CHECK_GT(info->input_layer.size(), 0);
        int pred_id = info->input_layer[0];
        auto info_p = collection[pred_id - 1];
        if (!(info_p->output_size[2] == 1 &&
            info_p->output_size[1] == 1)) {
          continue;
        }
        int64_t Hi = info_p->input_size[1];
        int64_t Wi = info_p->input_size[2];
        int64_t Kh = info_p->ker_size[1];
        int64_t Kw = info_p->ker_size[2];
        int64_t Sh = info_p->ker_stride[1];
        int64_t Sw = info_p->ker_stride[2];
        int64_t Ho = (Hi - Kh + 1) / Sh;
        int64_t Wo = (Wi - Kw + 1) / Sw;
        if (info_p->pooling_type != 0) {
          Kh = info_p->pooling_size[1];
          Kw = info_p->pooling_size[2];
          Sh = info_p->pooling_stride[1];
          Sw = info_p->pooling_stride[2];
          if (info_p->pooling_type == 2) {
            Ho = 1;
            Wo = 1;
          } else {
            Ho = (Hi - Kh + 1) / Sh;
            Wo = (Wi - Kw + 1) / Sw;  
          }
        }
        int64_t Co = info->input_size[3] / (Ho * Wo);
        info_p->output_size = {info_p->output_size[0], Ho, Wo, Co};
        info->input_size = {info_p->output_size[0], Ho, Wo, Co};
      }
    }
  }
  
  void CanonicalizeResidual() {
    for (auto info : collection) {
      // residual_source
      if (info->residual_source.size() != 0) {
        for (auto res_input_idx : info->residual_source) {
          // remove residual input from input_layer and corresponding output_layer
          // make input/output_layer is only for concatenation
          auto it = std::find(info->input_layer.begin(), info->input_layer.end(), res_input_idx);
          info->input_layer.erase(it);
          // update output
          OpuInfo* oinfo = collection[res_input_idx - 1];
          it = std::find(oinfo->output_layer.begin(), oinfo->output_layer.end(), info->index);
          oinfo->output_layer.erase(it);
          // update its output choice
          oinfo->output_choice[1] = 3;
        }
      }
    }
  }
  
  void SkipPoolPadding() {
    auto info = collection[0];
    int64_t pool_pad = std::accumulate(
        info->pool_padding_size.begin(),
        info->pool_padding_size.end(),
        0);
    if (pool_pad > 0) {
        int64_t pool_x = (info->output_size[1] - info->post_padding_size[0] - info->post_padding_size[1] - 1)
            * info->pooling_stride[1] + info->pooling_size[1];
        int64_t pool_y = (info->output_size[2] - info->post_padding_size[2] - info->post_padding_size[3] - 1)
            * info->pooling_stride[2] + info->pooling_size[2];
        info->input_size[1] = (pool_x - 1) * info->ker_stride[1] + info->ker_size[1];
        info->input_size[2] = (pool_y - 1) * info->ker_stride[2] + info->ker_size[2];
        info->pool_padding_size = {0,0,0,0};
    }
  }
  
  void ToPostPadding() {
    CHECK_EQ(collection[0]->padding_size.size(), 4);
    post_padding = true;
    for (auto info : collection) {
      info->post_padding_mode = 1;
      for (size_t j = 0; j < info->padding_size.size(); j++) {
        info->post_padding_size.push_back(0);
        info->pre_remove_padding_size.push_back(0);
      }
    }
    // back propagate pre-padding(Boolean) to post-padding
    std::unordered_map<int, bool> need_post_padding;
    for (auto info : collection) {
      int sum = 0;
      for (auto x : info->padding_size)
        sum += x;
      // pad to input size NHWC
      if (info->index != 0) {
        info->input_size[1] += info->padding_size[0] + info->padding_size[1];
        info->input_size[2] += info->padding_size[2] + info->padding_size[3];
      }
      if (sum > 0) {
        for (auto iid : info->input_layer) {
          need_post_padding[iid] = true;
          if (iid == 0) {
            collection[0]->extra_pre_padding = 1;
          } else {
            collection[iid - 1]->post_padding = 1;
            for (size_t j = 0; j < info->padding_size.size(); j++) {
              collection[iid - 1]->post_padding_size[j] = info->padding_size[j];
            }
          }
        }
      }
    }
    // check concat inputs (should have same post padding)
    for (auto info : collection) {
      if (info->input_layer.size() > 1) {
        int64_t pp_idx = -1;
        for (auto id : info->input_layer) {
          if (collection[id - 1]->post_padding == 1) {
            pp_idx = id;
            break;
          }
        }
        if (pp_idx != -1) {
          for (auto id : info->input_layer) {
            collection[id - 1]->post_padding = 1;
            for (size_t j = 0; j < info->padding_size.size(); j++) {
              collection[id - 1]->post_padding_size[j] = collection[pp_idx - 1]->post_padding_size[j];
            }
          }  
        }
      }
    }
    
    // some layers output for future residue and need post padding at the same time
    // assumption: the output for residue must be without padding
    // we need to check the layer using this residue
    // and further pad its input
    std::unordered_map<int, bool> need_for_future_residue;
    /*for (auto info : collection) {   
      if (info->residual_source.size() > 0) { 
        int res_iid = info->residual_source[0];
        auto it = need_post_padding.find(res_iid);
        if (it != need_post_padding.end()) {
          std::cout << "######### @" << info->index
          << " layer" << res_iid << " outputs for future residue and needs post padding\n";
          int kh = info->ker_size[1];
          int kw = info->ker_size[2];
          for (auto iid : info->input_layer) {
            std::cout << "layer " << iid << " post padding size [";
            for (auto a : collection[iid-1]->post_padding_size) std::cout << a << ",";
            std::cout << "] -> ";
            collection[iid - 1]->post_padding_size[0] += kh / 2;
            collection[iid - 1]->post_padding_size[1] += kh / 2;
            collection[iid - 1]->post_padding_size[2] += kw / 2;
            collection[iid - 1]->post_padding_size[3] += kw / 2;
            std::cout << "[";
            for (auto a : collection[iid-1]->post_padding_size) std::cout << a << ",";
            std::cout << "]\n";
          }
          std::cout << "pre padding size [";
          for (auto a :info->padding_size) std::cout << a << ",";
          std::cout << "]\n";
          info->padding_size[0] += 1;//kh / 2;
          info->padding_size[1] += 1;//kh / 2;
          info->padding_size[2] += kw / 2;
          info->padding_size[3] += kw / 2;
          std::cout << " -> [";
          for (auto a :info->padding_size) std::cout << a << ",";
          std::cout << "]\n";
          need_for_future_residue[info->index] = true;
        }
      }
    }*/
    // assume symmetric post padding size is a hard constraint
    // check pre_remove_padding_size
    for (auto info : collection) {
      if (info->post_padding == 1) {
        
        bool symmetric = (
            (info->post_padding_size[0] == info->post_padding_size[1]) &&
            (info->post_padding_size[2] == info->post_padding_size[1])
            )?
            true : false;
        if (!symmetric) {
          std::cout << "layer " << info->index << " unsymmetric post padding\n";
          std::cout << "[";
          for (auto& x : info->post_padding_size) {
            std::cout << x << ",";
          }std::cout << "]";
          auto max_iter = std::max_element(
            info->post_padding_size.begin(),
            info->post_padding_size.end());
          int64_t max = *max_iter;
          for (auto& x : info->post_padding_size) {
            x = max;
          }
          std::cout << "->[";
          for (auto& x : info->post_padding_size) {
            std::cout << x << ",";
          }std::cout << "]\n";
        }
        
        for (auto oid : info->output_layer) {
          for (size_t j = 0; j < info->padding_size.size(); j++) {
            collection[oid - 1]->pre_remove_padding_size[j] =
                info->post_padding_size[j] -
                collection[oid - 1]->padding_size[j];
          } 
        }
      }
    }
    // update output size
    for (auto info : collection) {
      if (info->post_padding == 1) {
        info->output_size[1] += info->post_padding_size[0] + info->post_padding_size[1];
        info->output_size[2] += info->post_padding_size[2] + info->post_padding_size[3];
      }
    }
    for (auto info : collection) {
      for (auto iid : info->input_layer) {
        if (iid != 0) {  // NHWC
          info->input_size[1] = collection[iid - 1]->output_size[1];
          info->input_size[2] = collection[iid - 1]->output_size[2];
        }
      }
    }
    
    for (auto item : need_for_future_residue) {
      auto info = collection[item.first - 1];
      info->post_padding = 0;
      for (size_t j = 0; j < info->padding_size.size(); j++) {
        info->post_padding_size[j] = 0;
      }
    }
  }

  void dump2json(std::string filename) {
    std::ofstream outj(filename);
    // dump OpuInfo to json one by one
    for (auto info : collection) {
      json j;
      info->to_json(j);
      outj << j << "\n";
    }
    outj.close();
    // code to parse dumped json from above
    /*
    std::ifstream inj("opu_ir.json");
    std::string line;
    while (std::getline(inj, line)) {
      std::istringstream iss(line);
      json jt;
      iss >> jt;
      OpuInfo* info = new OpuInfo();
      info->from_json(jt);    
    }
    inj.close();
    */
  }
  
  void dump2file() {
    std::ofstream out;
    out.open("OPU_IR_Readable.txt");
    std::ostringstream os;
    for (auto info : collection) {
       info->dump(os);
    }
    out << os.str();
    out.close();

    std::ostringstream os1;
    this->dump_decr(os1);
    out.open("OPU_IR.txt");
    out << os1.str();
    out.close();
    
    dump2json("OPU_IR.json");
    
    std::ostringstream os2;
    os2 << "\n";
    os2 << "============================\n";
    os2 << "Successfully Generate OPU IR\n";
    os2 << "-> OPU_IR_Readable.txt\n";
    os2 << "-> OPU_IR.txt\n";
    os2 << "-> OPU_IR.json\n";
    os2 << "============================\n";
    LOG(INFO) << os2.str();
  }

  void dump(std::ostringstream& os) {
    os << "num: [" << collection.size() << "]\n";
    os << "type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "input_layer:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->input_layer;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_layer:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_layer;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "input_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->input_size;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_size;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "ker_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->ker_size;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "ker_stride:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->ker_stride;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pooling_size;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling_stride:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pooling_stride;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling_type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->pooling_type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pool_padding_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pool_padding_size;
       for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
     }
     os << "]";
     if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "padding_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->padding_size;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "activation_type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->activation_type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "residual_source:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->residual_source;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_choice:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_choice;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
       }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "res_position:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->res_position;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "group:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->group;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "channel_shuffle:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->channel_shuffle;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "upsample:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->upsample;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    if (quantized) {
      os << "input_fm_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->input_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "output_fm_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->output_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "weights_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->weight_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "bias_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->bias_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    }    
  }
  
  void dump_decr(std::ostringstream& os) {
    os << "num: [" << collection.size() << "]\n";
    os << "type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "input_ddr_addr:[";  // input_layer
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->input_layer;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_ddr_addr:[";  // output_layer
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_layer;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "input_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->input_size;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_size;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "ker_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->ker_size;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "," << u[0];
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "ker_stride:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->ker_stride;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "," << u[0];
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling_size:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pooling_size;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "," << u[0];
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling_stride:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pooling_stride;
      for (size_t j=1; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "," << u[0];
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "pooling:[";  // none
    for (size_t i = 0; i < collection.size(); i++) {
      if (collection[i]->pooling_type == 0) {
        os << "0";
      } else {
        os << "1";
      }
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";
    
    os << "pooling_type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->pooling_type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";
    
    os << "pool_pad:[";  // none
    for (size_t i = 0; i < collection.size(); i++) {
      std::vector<opu_int> u = collection[i]->pool_padding_size;
      int sum = 0;
      for (size_t j=0; j < u.size(); j++) {
        sum += u[j];
      }
      if (sum == 0) {
        os << "0";
      } else {
        os << "1";
      }
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";
    
    os << "pool_pad_size:[";  // pool_padding_size
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->pool_padding_size;
       for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
     }
     os << "]";
     if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";
    
    if (post_padding) {
      os << "post_padding:[";
      for (size_t i = 0; i < collection.size(); i++) {
        if (collection[i]->post_padding == 0) {
          os << "0";
        } else {
          os << "1";
        }
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    
      os << "post_padding_size:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << "[";
        std::vector<opu_int> u = collection[i]->post_padding_size;
        for (size_t j=0; j < u.size(); j++) {
          os << u[j];
          if (j != u.size()-1) {
            os << ",";
          }
        }
        os << "]";
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    
      os << "pre_remove_padding:[";  // none
      for (size_t i = 0; i < collection.size(); i++) {
        os << "[";
        std::vector<opu_int> u = collection[i]->pre_remove_padding_size;
        for (size_t j=0; j < u.size(); j++) {
          os << u[j];
          if (j != u.size()-1) {
            os << ",";
          }
        }
        os << "]";
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    } else {
      os << "pre_padding:[";  // none
      for (size_t i = 0; i < collection.size(); i++) {
        std::vector<opu_int> u = collection[i]->padding_size;
        int sum = 0;
        for (size_t j=0; j < u.size(); j++) {
          sum += u[j];
        }
        if (sum == 0) {
          os << "0";
        } else {
          os << "1";
        }
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    
      os << "pre_padding_size:[";  // padding_size
      for (size_t i = 0; i < collection.size(); i++) {
        os << "[";
        std::vector<opu_int> u = collection[i]->padding_size;
        for (size_t j=0; j < u.size(); j++) {
          os << u[j];
          if (j != u.size()-1) {
            os << ",";
          }
        }
        os << "]";
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    
      os << "pre_remove_padding:[";  // none
      for (size_t i = 0; i < collection.size(); i++) {
        os << "[";
        std::vector<opu_int> u = collection[i]->padding_size;
        for (size_t j=0; j < u.size(); j++) {
          os << "0";
          if (j != u.size()-1) {
            os << ",";
          }
        }
        os << "]";
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n"; 
    }

    os << "activation_type:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->activation_type;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "residual:[";  // none
    for (size_t i = 0; i < collection.size(); i++) {
      std::vector<opu_int> u = collection[i]->residual_source;
      if (u.size() == 0) {
        os << "0";
      } else {
        os << "1";
      }
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";
    
    os << "shortcut_source:[";  // residual_source
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->residual_source;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
        }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "output_choice:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << "[";
      std::vector<opu_int> u = collection[i]->output_choice;
      for (size_t j=0; j < u.size(); j++) {
        os << u[j];
        if (j != u.size()-1) {
          os << ",";
       }
      }
      os << "]";
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "res_position:[";
    for (size_t i = 0; i < collection.size(); i++) {
      int res = collection[i]->res_position;
      if (res == 0) {
        os << "0";
      } else {
        os << res;
      }
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "group:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->group;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "channel_shuffle:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->channel_shuffle;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    os << "upsampling_scale:[";
    for (size_t i = 0; i < collection.size(); i++) {
      os << collection[i]->upsample;
      if (i != collection.size()-1) {
        os << ",";
      }
    }
    os << "]\n";

    if (quantized) {
      os << "input_fm_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->input_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "output_fm_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->output_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "weights_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->weight_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
      
      os << "bias_fix_lo:[";
      for (size_t i = 0; i < collection.size(); i++) {
        os << collection[i]->bias_fraclen;
        if (i != collection.size()-1) {
          os << ",";
        }
      }
      os << "]\n";
    }    
  }
};
}  // namespace relay
}  // namespace tvm
#endif  // SRC_RELAY_PASS_HW_INFO_H_
