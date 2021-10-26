import numpy as np
import time
import threading
import tensorflow as tf
import tvm
from .op import * 
#from .fp import fp as fixpoint
import scipy.io
from collections import defaultdict
from . import suppress_stdout_stderr as block
import cv2
import math
import copy
from .outWriter import outWriter
from .analysis import Analysis
from .debug import Debugger
import ipdb
import json
from .quantization_utils import Qnn

class simGen:
    def __init__(self, codeGenNodes, params, input_shape, fx_output_dir, configs):
        self.preprocess_method = configs['preprocess']
        self.input_shape = list(input_shape)[1:] # skip N
        self.configs = configs
        self.fx_output_dir = fx_output_dir 
        if self.fx_output_dir is not None:
            if not os.path.exists(self.fx_output_dir): 
                os.makedirs(self.fx_output_dir)            
            if os.path.exists(self.fx_output_dir+'fracLen.txt'): os.remove(self.fx_output_dir+'fracLen.txt')
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.tempNodes = []
        self.params = params
        self.followOps = ['nn.relu','nn.leaky_relu','nn.max_pool2d','nn.max_pool3d','nn.avg_pool2d','nn.avg_pool3d','add','clip','tanh','multiply','strided_slice','sigmoid','nn.global_avg_pool2d','concatenate','mean','divide','expand_dims','squeeze','strided_slice','transpose','reshape','nn.upsampling']
        self.nodes = codeGenNodes
        '''node = [x for x in self.nodes if x.attrs['id']==75][0]
        node.inputs = [73]
        node = [x for x in self.nodes if x.attrs['id']==74][0]
        self.nodes.remove(node)'''
        self.clearZeroPad(self.nodes)
        self.eliminate_concat_for_route(self.nodes)
        #self.simplify_hswish(self.nodes)
        self.pad=False
        self.var = tf.constant([0])
        self.fracLen = 0
        
        self.inp_data_np = self.readInput()
        if self.configs['dump']:
            np.save(os.path.join(self.fx_output_dir,'input'),self.inp_data_np)
        self.inp = self.inp_data_np
        self.layerId = 0
        self.varDict = defaultdict(dict)
        self.cut_pos_fl = 0
        self.ctpDict = {} # store cut pos fracLen based on TF results for each layer's output!
        self.nodeLayerIdMap = {}
        self.inpLayerId=-1
        self.ctpDictConcat={}
        self.layerNodes = []
        self.input_quantize_layerId_map={}
        self.input_node_ids = []
        #self.target = ["sw"]#, "opu"]
        self.target = ["sw"] if not configs['gen_fx_pb'] else ['hw']
        self.dummy_input = True if self.preprocess_method is None else False
        self.single_add_layer=False # focus on the case where add in bn is not folded into conv bias
        self.focus_add_in_bn=False
        self.gen_pb =  configs['gen_pb'] #True 
        self.gen_fx_pb = configs['gen_fx_pb'] #False#True
        self.fm_wordLen = 8
        self.output_layer_param_names = defaultdict(list)
        self.layer_fracLen_dict = {'FM':{},'WEIGHT':{},'BIAS':{}}
        self.debugger = Debugger()
        self.dtype = 'fixedpoint' #'fp_opt' #'fp4e3' #'fixedpoint'
        self.w_dtype = 'fixedpoint' #'fp_opt'
        self.qnn = Qnn()
        '''for idx, node in enumerate(self.nodes):
            if 'id' not in node.attrs:
                node.attrs['id'] = -idx
            if node.attrs['id'] == 90:
                df = idx
        nnn = []
        for idx, node in enumerate(self.nodes):
            if node.attrs['id'] == 134:
                node.inputs=[89,89,89,89,132]
            if node.attrs['id'] <0:
                continue
            if not idx==df: nnn.append(node)
            if node.attrs['id'] == 82:
               nnn.append(self.nodes[df])     
        self.nodes = nnn'''
        
    def simplify_hswish(self, nodes):
        divide_nodes = [x for x in nodes if 'op_name' in dir(x) and x.op_name=='divide']
        div_hswish = []
        for item in divide_nodes:
            succ = [x for x in nodes if 'inputs' in dir(x) and 'op_name' in dir(x) and item.attrs['id'] in x.inputs][0]
            if succ.op_name=='multiply':
                div_hswish.append([item, succ])
        for div,fm_mul in div_hswish:
            # graph visit
            relu6_id = div.inputs[0]
            relu6 = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==relu6_id][0]
            add3_id = relu6.inputs[0]           
            add3 = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==add3_id][0]
            addbias_id = add3.inputs[0]
            bias3_id = add3.inputs[1]
            addbias = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==addbias_id][0]
            if addbias.op_name=='add':  # conv-add-hswish
                continue
                conv_id = addbias.inputs[0]
                bias_id = addbias.inputs[1]
                conv = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==conv_id][0]
                weight_id = conv.inputs[1]
                # graph mutate
                self.params[weight_id] = tvm.ndarray.array(self.params[weight_id].asnumpy()/np.sqrt(6))
                self.params[bias_id] = tvm.ndarray.array(self.params[bias_id].asnumpy()/np.sqrt(6))

            elif addbias.op_name=='multiply': # div(hsigmod)-fm_mul(outside hswish)-hswish
                #continue
                reshape_id = addbias.inputs[1]
                reshape = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==reshape_id][0]
                # hsigmoid
                div_hsigmoid_id = reshape.inputs[0]
                div_hsigmoid = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==div_hsigmoid_id][0]
                div6_id = div_hsigmoid.inputs[1]
                self.params[div6_id] = tvm.ndarray.array(self.params[div6_id].asnumpy()/6*14.75)#*np.sqrt(6))
                '''
                relu6_hs_id = div_hsigmoid.inputs[0]
                relu6_hs = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==relu6_hs_id][0]
                add3_hs_id = relu6_hs.inputs[0]
                add3_hs = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==add3_hs_id][0]
                bias3_hs_id = add3_hs.inputs[1]
                nodes[relu6_hs_id].op_attrs['max'] = np.sqrt(6)
                self.params[bias3_hs_id] = tvm.ndarray.array(self.params[bias3_hs_id].asnumpy()/np.sqrt(6))
                self.params[bias3_hs_id-3+1] = tvm.ndarray.array(self.params[bias3_hs_id-3+1].asnumpy()/np.sqrt(6))
                self.params[bias3_hs_id-10+1] = tvm.ndarray.array(self.params[bias3_hs_id-10+1].asnumpy()/np.sqrt(6))
                reshape.inputs[0] = relu6_hs_id
                # pred conv
                pred_biasadd_id = addbias.inputs[0]
                pred_biasadd = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==pred_biasadd_id][0]
                pred_bias_id = pred_biasadd.inputs[1]
                pred_conv_id = pred_biasadd.inputs[0]
                pred_conv = [x for x in nodes if 'op_name' in dir(x) and x.attrs['id']==pred_conv_id][0]
                pred_weight_id = pred_conv.inputs[1]
                self.params[pred_weight_id] = tvm.ndarray.array(self.params[pred_weight_id].asnumpy()/np.sqrt(6))
                self.params[pred_bias_id] = tvm.ndarray.array(self.params[pred_bias_id].asnumpy()/np.sqrt(6))
                '''
            self.params[bias3_id] = tvm.ndarray.array(self.params[bias3_id].asnumpy()/np.sqrt(6))
            nodes[relu6_id].op_attrs['max'] = np.sqrt(6)   
            idx = fm_mul.inputs.index(div.attrs['id'])
            fm_mul.inputs[idx] = relu6_id
            #import ipdb
            #ipdb.set_trace()
            #print()
    
    def eliminate_concat_for_route(self, nodes):
        concat_nodes = [x for x in nodes if 'op_name' in dir(x) and x.op_name=='concatenate']
        route_nodes = []
        for node in concat_nodes:
            num_input = len(node.inputs)
            if num_input == 1:
                pred_id = node.inputs[0]
                cur_id = node.attrs['id']
                for succ in nodes:
                    if 'inputs' in dir(succ) and cur_id in succ.inputs:
                        idx = succ.inputs.index(cur_id)
                        succ.inputs[idx] = pred_id
                route_nodes.append(node)
        for node in route_nodes:
            nodes.remove(node)
        
        
    def merge_upsampling_concat_conv(self, nodes):
        #upsampling_node_ids = [ii for ii in range(len(nodes)) if nodes[ii].op_name=='nn.sampling']
        pass
    
    def clearZeroPad(self, nodes):
        pad_nodes = [x for x in nodes if 'op_name' in dir(x) and x.op_name=='nn.pad']
        def pad_sum(pad_width):
            ps = [sum([t.value for t in x]) for x in pad_width]
            return sum(ps)
        zeropad_nodes = [x for x in pad_nodes if pad_sum(x.op_attrs['pad_width'])==0]
        zp_ids = [x.attrs['id'] for x in zeropad_nodes]
        maps = {}
        for zp in zeropad_nodes:
            id = zp.attrs['id']
            iid = zp.inputs[0]
            maps[id] = iid
        def minimize(maps):
            for x in maps.keys():
                y = maps[x]
                while y in maps.keys():
                    y = maps[y]
                maps[x] = y
            return maps
        maps = minimize(maps)
        for zp in zeropad_nodes:
            id = zp.attrs['id']
            outs = [x.attrs['id'] for x in nodes if 'inputs' in dir(x) and id in x.inputs]
            for oid in outs:
                new_inputs = []
                for x in nodes[oid].inputs:
                    if x==id:
                        new_inputs.append(maps[x])
                    else:
                        new_inputs.append(x)
                nodes[oid].inputs = new_inputs
        for zp in zeropad_nodes:
            nodes.remove(zp)
           
            
            
                

    def readInput(self):
        if self.preprocess_method is None:
            return np.expand_dims(np.random.normal(0, 1, self.input_shape),0).astype(np.float32)
            ### input for vehicle detection model (yolov2 with reorg removed) 
            '''dat = scipy.io.loadmat("/home/tiandong/Downloads/tempD/inps/inp_vehicle_detection_model.mat") 
            inp_data_np = dat['value'][:,:,:,::-1]
            return np.expand_dims(inp_data_np[0],0)'''
        elif self.preprocess_method=='lp_detect':
            ### input for lp detection model (fast yolo) 
            dat = scipy.io.loadmat("/home/tiandong/Downloads/tempD/inps/inp_lp_detection_model.mat")
            inp_data_np = dat['value'][:,:,:,::-1]
            return np.expand_dims(inp_data_np[0],0)
            ### input for character detection model 
            '''dat = scipy.io.loadmat("/home/tiandong/Downloads/tempD/inps/inp_character_recognition_model.mat") 
            inp_data_np = dat['value']
            return np.expand_dims(inp_data_np[0],0)'''
        ### input from img
        elif self.preprocess_method=='inception':
            from .preprocess.inception_preprocessing import preprocess_image
            inp_image_path = self.configs['ref_input'] #'/home/tiandong/Downloads/imagenet/ILSVRC2012_val_00001110.JPEG'
            inp_data_np = cv2.imread(inp_image_path)
            inp_data_np = inp_data_np[:,:,::-1]
            inp_data_np = preprocess_image(tf.constant(inp_data_np), self.input_shape[0], self.input_shape[1])
            inp_data_np = self.sess.run(inp_data_np)
            return np.expand_dims(inp_data_np,0)
        ### mobilenet
        elif self.preprocess_method=='tf_mobilenet':
            from PIL import Image
            inp_image_path = self.configs['ref_input'] #'/home/tiandong/Downloads/imagenet/ILSVRC2012_val_00001110.JPEG'
            inp_data_np = cv2.imread(inp_image_path)
            inp_data_np = inp_data_np[:,:,::-1]
            img = Image.fromarray(inp_data_np,'RGB')
            x = np.array(img.resize((self.input_shape[0], self.input_shape[1]))).astype(np.float) / 128 - 1
            x = np.expand_dims(x,0)
            return x
        ### input from img PyTorch preprocess
        elif self.preprocess_method=='torchvision':
            from torchvision import transforms
            import torch
            transform = transforms.Compose([            #[1]
             transforms.Resize(256),                    #[2]
             transforms.CenterCrop(self.input_shape[0]),                #[3]
             transforms.ToTensor(),                     #[4]
             transforms.Normalize(                      #[5]
             mean=[0.485, 0.456, 0.406],                #[6]
             std=[0.229, 0.224, 0.225]                  #[7]
             )])
            from PIL import Image
            inp_image_path = self.configs['ref_input'] #'/home/tiandong/Downloads/imagenet/ILSVRC2012_val_00001110.JPEG'
            inp_data_np = cv2.imread(inp_image_path)
            inp_data_np = inp_data_np[:,:,::-1]
            img = Image.fromarray(inp_data_np,'RGB')
            img_t = transform(img)
            x = torch.unsqueeze(img_t, 0)
            x = x.detach().numpy().transpose(0,2,3,1)
            return x
        ### custuomzied input i.e. inception_v2 after 1st depthwise conv2d
        #inp_data_np = np.load('inception_v2_input.npy')
        #return inp_data_np.astype(np.float32)
        # vgg16 imagenet
        elif self.preprocess_method=='vgg16':
            inp_image_path = '/home/tiandong/Downloads/imagenet/ILSVRC2012_val_00000010.JPEG'
            inp_data_np = cv2.imread(inp_image_path)
            inp_data_np = inp_data_np[:,:,::-1]
            inp_data_np = inp_data_np.astype(np.float32)
            mean = np.load('/home/tiandong/Downloads/imagenet_unofficial/ilsvrc_2012_mean.npy')
            inp_data_np = cv2.resize(inp_data_np,(256,256))# - mean.transpose(1,2,0)         
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            ch_r = inp_data_np[:,:,2:3]-_R_MEAN
            ch_g = inp_data_np[:,:,1:2]-_G_MEAN
            ch_b = inp_data_np[:,:,0:1]-_B_MEAN
            inp_data_np = np.concatenate((ch_b,ch_g,ch_r),axis=2)
            inp_data_np = cv2.resize(inp_data_np,(self.input_shape[0], self.input_shape[1]))/255.
            return np.expand_dims(inp_data_np.astype(np.float32),0)
        # vgg19 keras
        elif self.preprocess_method=='vgg19_keras':
            from keras.preprocessing import image
            from keras.applications.vgg19 import preprocess_input
            img_path = self.configs['ref_input'] #'/home/tiandong/Downloads/imagenet/ILSVRC2012_val_00000010.JPEG'
            img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x
        # customzied
        elif self.preprocess_method=='custom':
            #tmp = np.load('C3D/inp.npy')
            #tmp = np.load('I3D/I3D-Tensorflow/ucf101_sample.npy')
            #tmp = np.load('S3D/s3d.pytorch/inp.npy').transpose(0,2,3,4,1)[:,0,:,:,:]
            #tmp = np.load('P3D/p3d.pytorch/inp.npy').transpose(0,2,3,4,1)
            #tmp = np.load('yolov3/PyTorch-YOLOv3/yolov3_inp.npy')
            #tmp = np.load('yolov2/yolov2_inp.npy')
            tmp = np.load(self.configs['ref_input'])
            return tmp
        # dcgan generator -- input [None, 100]
        #return np.load('inp_zero.npy')
        #return np.random.normal(0, 1, (1, 100))
        # discogan
        '''tmp = np.load('/home/tiandong/tvm/example/tests/discogan/imgs_A.npy')
        tmp = tmp.transpose(0,3,1,2).astype(np.float32)
        return tmp[0:1,:,:,:]'''
        # unet
        '''tmp = np.load('inp.npy')
        tmp = tmp.transpose(0,3,1,2).astype(np.float32)
        return tmp[0:1,:,:,:]'''
        # artgan
        '''tmp = np.load('/home/tiandong/Downloads/transpose_test/ArtGAN/ArtGAN/z_np.npy')
        return tmp'''
        
        
    def setSaveDir(self, path):
        self.savePath = path
    
    def split_scale_in_mobilenetv3(self):
        scale_node = None
        idx = None
        for i,node in enumerate(self.tempNodes):
            if node.op_name=='multiply':
                ids = [x.attrs['id'] for x in self.nodes if isinstance(x,tvm.relay.backend.graph_runtime_codegen.OpNode)]
                is_input_fm = [1 if x in ids else 0 for x in node.inputs]
                if sum(is_input_fm)==len(node.inputs):
                    cur_ids = [x.attrs['id'] for x in self.tempNodes]
                    in_cur_layer = [1 if x in cur_ids else 0 for x in node.inputs]
                    if sum(in_cur_layer)==1:
                        scale_node = node
                        idx = i
            if scale_node is not None:
                if node.op_name in self.followOps:
                    idx = i
                else:
                    break
        main_node_id = [x.attrs['id'] for x in self.tempNodes if x.op_name in ['nn.conv2d','nn.cpnv3d','nn.dense']]
        if len(main_node_id)==0:
            main_node_id = -1
        else:
            main_node_id = main_node_id[0]
        if scale_node is not None:
            idx -= 1
            if idx>main_node_id:
                return copy.deepcopy([self.tempNodes[0:idx],self.tempNodes[idx:]])
            else:
                return copy.deepcopy([self.tempNodes[0:idx+1],self.tempNodes[idx+1:]])
        else:
            return [self.tempNodes]  
    
    
    def quantize_with_tf_sim(self):
        # if not 'compute_shape' in self.configs:
        node = self.nodes[0]
        if "input" in node.name:
            inp_shape = node.attrs['shape'][0][1:]
            inp_shape = self.input_shape
            self.ph = tf.placeholder(shape=[1]+inp_shape,dtype=tf.float32,name='input_fx')
            print('Model Input Shape:',inp_shape)
            # dummy input
            if self.gen_pb:
                self.inp = self.ph
            elif self.dummy_input:
                self.inp = np.expand_dims(np.random.normal(0, 1, inp_shape),0).astype(np.float32)
                #np.save('inp',self.inp)
                #self.inp = np.load('inp.npy')
            else:
                self.inp = self.inp_data_np
            nid = node.attrs['id']
            for mode in self.target:
                self.varDict[nid][mode] = self.inp
            self.nodeLayerIdMap[nid] = -1 
            print('node_id:',nid)
            self.input_node_ids.append(nid)
            assert inp_shape==list(self.inp.shape[1:]),"Input Shape Mismatch"
        ana = Analysis(self.nodes, self.params)
        for group in ana.groups:
            for x in group:
                self.tempNodes.append(self.nodes[x])
            print()
            print("="*10,"Layer",self.layerId,"="*10)
            self.run_sim(target=self.target)
            self.layerId += 1
            self.layerNodes.append(copy.deepcopy(self.tempNodes))
            self.tempNodes.clear()
        
        '''i = 0
        self.add_in_layer=False
        while i<len(self.nodes):
            node = self.nodes[i]
            if "input" in node.name:
                inp_shape = self.nodes[i].attrs['shape'][0][1:]
                inp_shape = self.input_shape
                self.ph = tf.placeholder(shape=[None]+inp_shape,dtype=tf.float32,name='input_fx')
                #if self.gen_pb: 
                    #inp_shape =[224,224,3]
                    #inp_shape = [16,160,160,3]
                print('Model Input Shape:',inp_shape)
                # dummy input
                if self.gen_pb:
                    self.inp = self.ph
                elif self.dummy_input:
                    self.inp = np.expand_dims(np.random.normal(0, 1, inp_shape),0).astype(np.float32)
                    #np.save('inp',self.inp)
                    #self.inp = np.load('inp.npy')
                else:
                    self.inp = self.inp_data_np
                nid = node.attrs['id']
                for mode in self.target:
                    self.varDict[nid][mode] = self.inp
                self.nodeLayerIdMap[nid] = -1 
                print('node_id:',nid)
                self.input_node_ids.append(nid)
                assert inp_shape==list(self.inp.shape[1:]),"Input Shape Mismatch"
                i+=1
                continue
            elif isinstance(node, tvm.relay.backend.graph_runtime_codegen.InputNode):
                i += 1
                continue
            elif node.name.startswith('p'):
                i+=1
                continue
            elif node.name.find('conv2d')!=-1 or node.name.find('dense')!=-1 or node.name.find('conv3d')!=-1:
                self.tempNodes.append(self.nodes[i])
                while True:
                    i+=1
                    if i>=len(self.nodes): break
                    if isinstance(self.nodes[i], tvm.relay.backend.graph_runtime_codegen.OpNode):
                        self.single_add_layer = self.nodes[i].op_name=='add' and self.add_in_layer
                        if 'op_name' in dir(self.nodes[i]):
                            if self.nodes[i].op_name=='nn.pad':
                                prev_ids = [x.attrs['id'] for x in self.tempNodes]
                                if not self.nodes[i].inputs[0] in prev_ids: break
                                tmp_i = i
                                i += 1
                                while self.nodes[i].name.startswith('p') or self.nodes[i].op_name=='transpose':
                                    i += 1
                                if self.nodes[i].op_name not in ['nn.conv2d','nn.conv3d','nn.dense']:
                                    self.tempNodes.append(self.nodes[tmp_i])
                                else:
                                    i = tmp_i
                                    break
                        # single avg_pool in inception_v3
                        if self.nodes[i].op_name in self.followOps and self.tempNodes[-1].attrs['id'] in self.nodes[i].inputs and not self.single_add_layer:
                            self.tempNodes.append(self.nodes[i])
                            if self.nodes[i].op_name=='add':self.add_in_layer=self.focus_add_in_bn
                        else:
                            break
                groups = self.split_scale_in_mobilenetv3()
                for group in groups:
                    self.tempNodes = group
                    print()
                    print("="*10,"Layer",self.layerId,"="*10)
                    self.run_sim(target=self.target)
                    self.layerId += 1
                    self.layerNodes.append(copy.deepcopy(self.tempNodes))
                    self.tempNodes.clear()
                    self.add_in_layer=False
                continue
            elif self.single_add_layer:
                print("="*10,"Layer",self.layerId,"="*10)
                self.tempNodes.append(self.nodes[i])
                i+=1
                self.run_sim(target=self.target)
                self.layerId += 1
                self.layerNodes.append(copy.deepcopy(self.tempNodes))
                self.tempNodes.clear()
                self.single_add_layer=False
                continue
            else:
                self.tempNodes.append(self.nodes[i])
                if self.nodes[i].op_name=='add': self.add_in_layer=self.focus_add_in_bn
                # single avg/max_pool
                if 'pool' in self.nodes[i].op_name and len(self.tempNodes)<=2:
                    #print(self.nodes[i])
                    #import ipdb
                    #ipdb.set_trace()
                    print("="*10,"Layer",self.layerId,"="*10)
                    self.run_sim(target=self.target)
                    self.layerId += 1
                    self.layerNodes.append(copy.deepcopy(self.tempNodes))
                    self.tempNodes.clear()
                    self.add_in_layer=False 
            i+=1'''
        
        if self.gen_pb:
            self.out_sw = self.var
            '''tmp = scipy.io.loadmat(os.path.join(self.fx_output_dir,'hw_ofm_'+str(self.layerId)+'.mat'))['value']
            osw = self.sess.run(self.out_sw,{self.ph:self.inp_data_np})
            import ipdb
            ipdb.set_trace()'''
            if self.gen_fx_pb:
                self.out_sw = self.quantize_fm(self.out_sw)
            '''osw = self.sess.run(self.var,{self.ph:self.inp_data_np})
            tmp = scipy.io.loadmat(os.path.join(self.fx_output_dir,"hw_ofm_"+str(self.layerId)+'.mat'))['value']
            import ipdb
            ipdb.set_trace()'''
            tmp = tf.identity(self.out_sw, name='output_fx')
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.sess, # The session is used to retrieve the weights
                self.sess.graph.as_graph_def(), # The graph_def is used to retrieve the nodes 
                self.configs['out_names']#['output_fx']#,'BiasAdd_58','BiasAdd_66'] # The output node names are used to select the usefull nodes
            ) 
            if self.fx_output_dir is not None:
                pb_name = os.path.join(self.fx_output_dir, 'test.pb')
                # Finally we serialize and dump the output graph to the filesystem
                with tf.gfile.GFile(pb_name, "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                print('='*20)
                print('Tensorflow Protobuf File Saved To', pb_name)
                print('='*20)
        self.sess.close()
        if self.fx_output_dir is not None:
            self.writeNetArchitecture() 
        # output fraclen info to file
        self.writeFracLenInfo()            
        return self.out_sw

        
        
    def writeFracLenInfo(self):
        if not self.configs['dump']:
            return
        with open(os.path.join(self.fx_output_dir,'fracLen.json'),'w') as fp:
            json.dump(self.layer_fracLen_dict, fp)
        with open(os.path.join(self.fx_output_dir,'fixLen.txt'),'w') as f:
            fm_fl = []
            for item in sorted(self.layer_fracLen_dict['FM']):
                fm_fl.append(self.layer_fracLen_dict['FM'][item])
            print('fix_fm_pos_ori = [',end='',file=f)
            for i,item in enumerate(fm_fl):
                if i==len(fm_fl)-1:
                    print(item,end='];\n',file=f)
                else:
                    print(item,end=',',file=f)
            w_fl = []
            for item in sorted(self.layer_fracLen_dict['WEIGHT']):
                w_fl.append(self.layer_fracLen_dict['WEIGHT'][item])
            print('fix_weights_fi_pos = [',end='',file=f)
            for i,item in enumerate(w_fl):
                if i==len(w_fl)-1:
                    print(item,end='];\n',file=f)
                else:
                    print(item,end=',',file=f)
            b_fl = []
            for item in sorted(self.layer_fracLen_dict['BIAS']):
                b_fl.append(self.layer_fracLen_dict['BIAS'][item])
            print('fix_bias_fi_pos = [',end='',file=f)
            for i,item in enumerate(b_fl):
                if i==len(b_fl)-1:
                    print(item,end='];\n',file=f)
                else:
                    print(item,end=',',file=f)
                    
    def writeNetArchitecture(self):
        fm_shapeDict = {x:self.varDict[x][self.target[0]].shape for x in self.varDict.keys()}
        param_shapeDict = {x:self.params[x].shape for x in self.params.keys()}
        shapeDict = {**fm_shapeDict, **param_shapeDict}
        ow = outWriter(self.layerNodes,shapeDict,self.params)
        ow.parse()
        ow.setWriteDir(self.fx_output_dir)
        ow.write2file()
        ow.write2file_opu_ir()
        print('\n'*2,'='*30,sep='')
        print('OPU IR GENERATED SUCCESSFULLY')
        print('='*30,'\n'*2,sep='')
    
    def run_sim(self, target=["sw"]):

        if "sw" in target: 
            print("="*10,"tensorflow","="*10)
            for group_id,node in enumerate(self.tempNodes):
                self.addOp(node, mode="sw", group_id=group_id)
            self.dump2outputdir('fp32_ofm',self.var)
            self.out_sw = self.var


        if "hw" in target:
            print("="*10,"hw","="*10)
            for group_id,node in enumerate(self.tempNodes):
                self.addOp(node, mode="hw", group_id=group_id)
            self.out_hw = self.var
        
        '''debug'''
        # self.tempNodes, self.varDict, dumped weights/bias
            
        #import ipdb
        #ipdb.set_trace()
        #print(target)
    
    def gerrf(self, mat_name, sim_out):
        dat = scipy.io.loadmat(mat_name)
        dat = dat['value']
        dat = np.expand_dims(dat,0)
        print("err=",self.compute_error(sim_out,dat))
        return dat
    
    def gerrt(self, mat_name, sim_out):
        dat = scipy.io.loadmat(mat_name)
        dat = dat['value']
        temp = sim_out.transpose(0,3,1,2)# -> 1,84,20,1
        with tf.Session() as sess:
            out = tf.nn.softmax(temp,axis=1)
            out = sess.run(out)
        sim_out = out[0:1,:,:,:]
        print("err=",self.compute_error(sim_out,dat))
    
    def gerr(self, mat_name, sim_out):
        dat = scipy.io.loadmat(mat_name)
        dat = dat['value']
        dat = np.expand_dims(dat[0],0)
        print("err=",self.compute_error(sim_out,dat))
     
    def addOp(self, node, mode='sw', group_id=None): # mode="sw":for pure tensorflow run
        self.nodeLayerIdMap[node.attrs['id']] = self.layerId
        #get correct input
        #most op only has one input, multi-input node are specified in its self branch below
        inp_node_id = node.inputs[0] 
        if inp_node_id in self.input_node_ids:
            self.layer_input_quantize_en = True
            self.input_quantize_layerId_map[inp_node_id] = self.layerId
        else:
            self.layer_input_quantize_en = False
        if inp_node_id in self.varDict.keys():
            self.var = self.varDict[inp_node_id][mode]
            tid = self.nodeLayerIdMap[inp_node_id]
            if tid!=-1 and tid!=self.layerId:
                self.inpLayerId = tid
        '''if group_id == 0:
            with tf.Session() as sess:
                osw = sess.run(self.var,{self.ph:self.inp_data_np})
            tmp = scipy.io.loadmat(os.path.join(self.fx_output_dir,"hw_ofm_"+str(self.layerId)+'.mat'))['value']
            import ipdb
            ipdb.set_trace()'''
        if group_id==0 and self.gen_fx_pb:
            self.var = self.quantize_fm(self.var)
        print(node.attrs['id'],node.op_name,'inputs:',node.inputs)
        self.var = dimension_check(mode, self.var)
        if node.op_name == "nn.pad": #input could be from previous layers instead of only last one
            # make sure padding layout is the same as input data layout!!
            pad_width = node.op_attrs['pad_width']
            inp_shape = self.var.shape
            #assert len(pad_width)==4,'pad_width is not 4'
            if 'platform' in self.configs.keys():
                layout = self.configs['platform']
            else:
                layout = None
            # only check when ONNX(NCHW) or first layer
            #if len(pad_width)==4
            if len(pad_width)==4 and (layout is not None or self.layerId==0):
                if pad_width[2][0].value==pad_width[3][0].value and pad_width[2][1].value==pad_width[3][1].value:
                    pad_format = 'NCHW'
                else:
                    pad_format = 'NHWC'
                if inp_shape[2]==inp_shape[3] and not inp_shape[2]==inp_shape[1]:#in case H=W=C, assume NHWC !!
                    inp_format = 'NCHW'
                else:
                    inp_format = 'NHWC'
                if inp_format=='NHWC' and pad_format=='NCHW':
                    pad_width = [pad_width[0],pad_width[2],pad_width[3],pad_width[1]]
                elif inp_format=='NCHW' and pad_format=='NHWC':
                    pad_width = [pad_width[0],pad_width[3],pad_width[1],pad_width[2]]
            elif len(pad_width)==5 and (layout is not None or self.layerId==0): #NCDHW -> NDHWC
                if pad_width[0][0].value==pad_width[1][0].value and pad_width[0][1].value==pad_width[1][1].value and pad_width[1][0].value==0:
                    pad_format = 'NCDHW'
                else:
                    pad_format = 'NDHWC'     
                if inp_shape[3]==inp_shape[4]:
                    inp_format = 'NCDHW'
                else:
                    inp_format = 'NDHWC'
                if inp_format=='NCDHW' and pad_format=='NDHWC':
                    pad_width = [pad_width[0],pad_width[4],pad_width[1],pad_width[2],pad_width[3]]
                elif inp_format=='NDHWC' and pad_format=='NCDHW':
                    pad_width = [pad_width[0],pad_width[2],pad_width[3],pad_width[4],pad_width[1]]
            # tf.pad
            print('[PAD]:',pad_width)
            self.var = tfpad(mode, self.var, pad_width)  
        elif node.op_name == "nn.conv2d":
            if node.inputs[1] in self.params.keys():
                weight = self.params[node.inputs[1]].asnumpy()
            else:
                weight = self.varDict[node.inputs[1]][mode]
            if mode=="sw":
                self.primeIn_sw = self.var 
            if 0 in node.inputs and self.var.shape[2]==self.var.shape[3]: 
                inp_data_format = 'NCHW'
            else:
                inp_data_format = 'NHWC'
            # padding
            if 'shape' in node.attrs.keys():
                out_shape = node.attrs['shape'][0]
                if node.op_attrs['data_layout']=='NCHW' and inp_data_format=='NHWC':
                    out_shape = [out_shape[0],out_shape[2],out_shape[3],out_shape[1]]
                output_shape = out_shape
                input_shape = self.var.shape
                if not isinstance(input_shape[0],int): input_shape = [x.value for x in input_shape]
                if inp_data_format=='NHWC':
                    if input_shape[1]%output_shape[1]==0 and input_shape[2]%output_shape[2]==0: 
                        padding = 'SAME'
                    else: 
                        padding = 'VALID'
                else:
                    if input_shape[3]%output_shape[3]==0 and input_shape[2]%output_shape[2]==0: # How to know padding mode directly from shape data?
                        padding = 'SAME'
                    else: 
                        padding = 'VALID'
            # kernel size and layout
            '''if self.layerId==36:
                import ipdb
                ipdb.set_trace() '''  
            kernel_format = node.op_attrs['kernel_layout']
            w_shape = weight.shape
            if not isinstance(w_shape[0], int):
                w_shape = [x.value for x in w_shape]
            if w_shape[1]==w_shape[2] and not w_shape[0]==w_shape[1]:
                kernel_format = 'OIHW'#'OHWI' # PyTorch :'OIHW'!!!!!!!!!!!!!!!!!!!!!!!!
            elif w_shape[0]==w_shape[1]:
                #if not w_shape[2]==w_shape[3]:
                #    kernel_format = 'HWOI' #T MobileNetV3 after SE
                #if not kernel_format=='OIHW':
                #    kernel_format = 'HWIO'
                x=1
            else:
                kernel_format = kernel_format[:2]+'HW'
            if kernel_format=='OHWI':
                kz_h,kz_w = weight.shape[1],weight.shape[2]
            elif kernel_format=='HWIO' or kernel_format=='HWOI':
                kz_h,kz_w = weight.shape[0],weight.shape[1]
            else:
                kz_h,kz_w = weight.shape[2],weight.shape[3]
            # stride
            if 'shape' in node.attrs:
                out_shape = node.attrs['shape'][0]
                if node.op_attrs['data_layout']=='NCHW' and inp_data_format=='NHWC':
                    out_shape = [out_shape[0],out_shape[2],out_shape[3],out_shape[1]]
                if output_shape[1]==0: # indicate transposed conv & output shape is determined by dilation size
                    stride_h, stride_w = 1, 1
                else:
                    if not isinstance(self.var.shape[1], int):
                        inp_shape = [x.value for x in self.var.shape]
                    else:
                        inp_shape = self.var.shape
                    if inp_data_format=='NHWC':
                        if padding=='VALID':
                            stride_h = self.getStride(inp_shape[1], out_shape[1], kz_h)
                            stride_w = self.getStride(inp_shape[2], out_shape[2], kz_w)
                        else:
                            stride_h = inp_shape[1]//out_shape[1]
                            stride_w = inp_shape[2]//out_shape[2]
                    elif inp_data_format=='NCHW':
                        if padding=='VALID':
                            stride_h = self.getStride(inp_shape[2], out_shape[2], kz_h)
                            stride_w = self.getStride(inp_shape[3], out_shape[3], kz_w)
                        else:
                            stride_h = inp_shape[2]//out_shape[2]
                            stride_w = inp_shape[3]//out_shape[3]
            else:
                stride_h = 1
                stride_w = 1
            data_format = inp_data_format #node.op_attrs['data_layout']
            if 'groups' in node.op_attrs.keys():
                groups = node.op_attrs['groups']
            else:
                groups = None
            node.op_attrs['padding']=padding
            if self.gen_fx_pb:
                weight, fl = qnn.apply(weight, 8)
                self.layer_fracLen_dict['WEIGHT'][self.layerId] = fl
                print('<<< weight',fl,'/ 8>>>')
                '''with block.suppress_stdout_stderr():
                    tmp = fixpoint(weight,wordLen=8,opt=True)
                    tmp = fixpoint(weight, wordLen=8, fracLen=tmp._fl ,roundMethod='round',dtype=self.w_dtype)
                    self.layer_fracLen_dict['WEIGHT'][self.layerId] = tmp._fl
                    weight = tmp._d_fp
                print('<<< weight',tmp._fl,'/ 8>>>')
                print('#'*10,'diff:',np.sum(np.square(data_q-weight)))'''
                #print(groups)
                #import ipdb
                #ipdb.set_trace()
                #import matplotlib.pyplot as plt
                #plt.hist(tmp.flatten().tolist(), bins = 'auto')
                #plt.hist(weight.flatten().tolist(), bins = 'auto')
                #plt.show()                
                
                
                #import ipdb
                #ipdb.set_trace()
            '''check layout'''
            if data_format=='NCHW':# ->NHWC
                if mode=='opu':
                    self.var = self.var.transpose(0,2,3,1)
                elif mode=='sw':
                    self.var = tf.transpose(self.var, perm=[0,2,3,1])
            if kernel_format=='OIHW':# ->HWIO
                if mode=='opu':
                    weight = weight.transpose(2,3,1,0)
                elif mode=='sw':
                    weight = tf.transpose(weight, perm=[2,3,1,0])
                elif mode=='hw':
                    weight = tf.transpose(weight, perm=[2,3,1,0])
            elif kernel_format=='OHWI':
                if mode=='opu':
                    weight = weight.transpose(1,2,3,0)
                elif mode=='sw':
                    weight = tf.transpose(weight, perm=[1,2,3,0])
                elif mode=='hw':
                    weight = tf.transpose(weight, perm=[1,2,3,0])
            elif kernel_format=='HWOI':
                if mode=='opu':
                    weight = weight.transpose(0,1,3,2)
                elif mode=='sw':
                    weight = tf.transpose(weight, perm=[0,1,3,2])
                elif mode=='hw':
                    weight = tf.transpose(weight, perm=[0,1,3,2])
            self.dump2outputdir('weight',weight)
            self.debugger.collect(self.layerId,'WEIGHT',weight)
            self.debugger.collect(self.layerId,'FM',self.var,feed_dict={self.ph:self.inp_data_np})
            platform = 'onnx' if 'platform' in self.configs.keys() else None
            if self.gen_fx_pb:
                il_fm = 7-self.layer_fracLen_dict['FM'][self.layerId] if self.layer_fracLen_dict['FM'][self.layerId]<=7 else 0
                il_w = 7-self.layer_fracLen_dict['WEIGHT'][self.layerId] if self.layer_fracLen_dict['WEIGHT'][self.layerId]<=7 else 0
                self.cut_pos_fl = 16-il_fm-il_w-1
            print('<',groups,'>',weight.shape)
            self.var= conv2d(mode, self.var, weight, [1,stride_h,stride_w,1], padding, data_format, kernel_format, groups, self.fracLen, self.layerId, self.cut_pos_fl, self.fx_output_dir, {self.ph: self.inp_data_np}, platform)
            self.debugger.collect(self.layerId,'IPA',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name== "nn.conv2d_transpose":
            if node.inputs[1] in self.params.keys():
                weight = self.params[node.inputs[1]].asnumpy()
            else:
                weight = self.varDict[node.inputs[1]][mode]
            output_shape = node.attrs['shape'][0]
            output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
            #print(output_shape)
            pad = [x.value for x in node.op_attrs['padding']]
            padding = 'SAME' if sum(pad)>0 else 'VALID'
            stride = node.op_attrs['strides'][0].value
            import ipdb
            ipdb.set_trace()
            self.var = conv2d_transpose(mode, self.var, weight.transpose(2,3,1,0),output_shape,stride,padding)
            import ipdb
            ipdb.set_trace()
        elif node.op_name == "add":
            if node.inputs[0] in self.varDict.keys() and node.inputs[1] in self.varDict.keys():
                print('residual_add')# residual add
                node.op_attrs['residual'] = True
                operand_0 = self.varDict[node.inputs[0]][mode]
                operand_1 = self.varDict[node.inputs[1]][mode]
                self.var = residualAdd(mode, operand_0, operand_1)
            else:
                #bias add
                bias = self.params[node.inputs[1]].asnumpy()   
                while bias.ndim>1:
                    if bias.shape[-1]==1:
                        bias = np.squeeze(bias,axis=bias.ndim-1)
                    else:
                        bias = np.squeeze(bias,axis=0)
                if bias.ndim==0: # broadcast add with single value
                    print('broadcast add with single value')
                    offset = bias.tolist()
                    in_channel = self.var.shape[-1].value
                    bias = np.array([offset for i in range(in_channel)],dtype=bias.dtype)
                    broadcast = True
                else:
                    broadcast = False
                if mode=='hw' and self.gen_fx_pb:
                    '''with block.suppress_stdout_stderr():
                        tmp = fixpoint(bias,wordLen=16,opt=True)
                        bias = fixpoint(bias, wordLen=16, fracLen=tmp._fl)._d_fp
                    print('<<< quantize bias',tmp._fl,'/16>>>')
                    self.layer_fracLen_dict['BIAS'][self.layerId] = tmp._fl'''
                    bias, fl = self.qnn.apply(bias, 16)
                    self.layer_fracLen_dict['BIAS'][self.layerId] = fl
                    print('<<< quantize bias',fl,'/16>>>')
                self.dump2outputdir('bias',bias)
                self.debugger.collect(self.layerId,'BIAS',bias)
                self.var = biasAdd(mode, self.var, bias, self.layerId, self.fx_output_dir)
                self.debugger.collect(self.layerId,'IPA',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "nn.leaky_relu":
            alpha = node.op_attrs['alpha']
            alpha = 0.125 if mode=='hw' else 0.1# hardware defined
            self.var = leakyRelu(mode, self.var, alpha)
            self.debugger.collect(self.layerId,'ACTIVATION',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "nn.relu": 
            self.var = relu(mode, self.var)
            self.debugger.collect(self.layerId,'ACTIVATION',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "clip":
            self.var = clip(mode, self.var, node.op_attrs['min'], node.op_attrs['max'])
            self.debugger.collect(self.layerId,'ACTIVATION',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "tanh": 
            self.var = tanh(mode, self.var)
        elif node.op_name == "sigmoid": 
            self.var = sigmoid(mode, self.var)
        elif node.op_name == "nn.max_pool2d":
            kz = [x.value for x in node.op_attrs['pool_size']]
            stride = [x.value for x in node.op_attrs['strides']]
            pad = node.op_attrs['padding']
            if sum(pad).value==0: 
                pad_mode='VALID'
            else: 
                pad_mode='SAME'
            platform = 'onnx' if 'platform' in self.configs.keys() else None
            self.var = maxPool(mode, self.var, [1,kz[0],kz[1],1], [1,stride[0],stride[1],1], pad_mode, 'NHWC', platform, pad)#node.op_attrs['layout'])
            self.debugger.collect(self.layerId,'POOLING',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "nn.avg_pool2d":
            kz = [x.value for x in node.op_attrs['pool_size']]
            stride = [x.value for x in node.op_attrs['strides']]
            pad = node.op_attrs['padding']
            if sum(pad).value==0: pad_mode='VALID'
            else: pad_mode='SAME'
            self.var = avgPool(target=mode, ifm=self.var, kz=[1,kz[0],kz[1],1], strides=[1,stride[0],stride[1],1], pad_mode=pad_mode, data_format='NHWC')#node.op_attrs['layout'])
            self.debugger.collect(self.layerId,'POOLING',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "nn.global_avg_pool2d":
            self.var = globalAvgPool(target=mode, ifm=self.var)
            self.debugger.collect(self.layerId,'POOLING',self.var,feed_dict={self.ph:self.inp_data_np})
        elif node.op_name == "concatenate":
            for item in node.inputs:
                if item not in self.varDict.keys():
                    for x in self.target:
                        self.varDict[item][x] = self.params[item].asnumpy()
            concat_inps = [self.varDict[x][mode] for x in node.inputs]
            axis = node.op_attrs['axis']
            if len(concat_inps[0].shape)==5:# NDHWC for 5d tensor
                axis = 4
            else:
                axis = 3
            self.var = concat(mode, concat_inps, axis)
        elif node.op_name=="transpose":
            if 'axes' in node.op_attrs:
                axes = [x.value for x in node.op_attrs['axes']]
                self.var = tf.transpose(self.var,perm=axes)
                #self.var = transpose(mode, self.var, axes, node.attrs['shape'][0])
            '''shape = [x.value for x in self.var.shape]
            ci = shape[-1]//3
            if ci==20:
                x = self.var
                xc = tf.concat([x[:,:,:,0:7],x[:,:,:,20:27],x[:,:,:,40:46],\
                                x[:,:,:,46:53],x[:,:,:,7:14],x[:,:,:,27:33],\
                                x[:,:,:,33:40],x[:,:,:,53:60],x[:,:,:,14:20]],axis=3)
                self.var = xc
            elif ci==40:
                x = self.var
                xc = tf.concat([x[:,:,:,0:14],x[:,:,:,40:53],x[:,:,:,80:93],\
                                x[:,:,:,53:67],x[:,:,:,93:106],x[:,:,:,14:27],\
                                x[:,:,:,106:120],x[:,:,:,27:40],x[:,:,:,67:80]],axis=3)
                self.var = xc
            elif ci==80:
                x = self.var
                xc = tf.concat([x[:,:,:,0:27],x[:,:,:,80:107],x[:,:,:,160:186],\
                                x[:,:,:,186:213],x[:,:,:,27:54],x[:,:,:,107:133],\
                                x[:,:,:,133:160],x[:,:,:,213:240],x[:,:,:,54:80]],axis=3)
                self.var = xc
            else:
                import ipdb
                ipdb.set_trace()'''
        elif node.op_name=="mean":
            oshape = [self.var.shape[0]]
            for i in range(1, len(node.attrs['shape'])+1):
                oshape.append(1) 
            self.var = globalAvgPool(target=mode, ifm=self.var)#self.var = mean(mode, self.var, oshape)
        elif node.op_name=='reshape':
            new_shape = [x.value for x in node.op_attrs['newshape']]
            if len(new_shape)==4:
                # assume new_shape as 'NHWC'
                if new_shape[2]==new_shape[3]:#indicate 'NCHW'
                    new_shape = [new_shape[0],new_shape[2],new_shape[3],new_shape[1]]
            platform = 'onnx' if 'platform' in self.configs.keys() else None        
            new_shape[0]=-1
            if new_shape[-1] == -1 and not 'compute_shape' in self.configs: # in case new_shape=[-1,56,56,-1]
                in_shape = [x.value for x in self.var.shape]
                tmp = 1
                for item in in_shape[1:]: tmp *= item
                if len(new_shape)==2:
                    new_shape[-1] = in_shape[-1]
                else:    
                    new_shape[-1] = tmp//(new_shape[2]*new_shape[1])
            if len(self.var.shape)==5:
                self.var = tf.transpose(self.var,perm=[0,3,4,1,2])
                print('tranpose',[0,3,4,1,2])
            self.var = reshape(mode, self.var, new_shape, platform)
        elif node.op_name=='nn.dense':
            weight = self.params[node.inputs[1]].asnumpy()
            if mode=="sw":
                self.primeIn_sw = self.var
            if mode=='hw':#self.gen_pb and self.gen_fx_pb:
                '''with block.suppress_stdout_stderr():
                    tmp = fixpoint(weight,wordLen=8,opt=True)
                    weight = fixpoint(weight, wordLen=8, fracLen=tmp._fl)._d_fp'''
                weight, fl = self.qnn.apply(weight, 8)
                print('<<< weight',fl,'/ 8>>>')
            self.dump2outputdir('weight',weight)
            self.var = dense(mode, self.var, weight, self.layerId, self.fx_output_dir)
        elif node.op_name=='nn.upsampling':
            scale = node.op_attrs['scale']
            self.var = upsampling(mode, self.var, scale)
        elif node.op_name=='image.resize':
            size = [x.value for x in node.op_attrs['size']]
            dilation_size = [x.value for x in node.op_attrs['dilation_size']]
            method = node.op_attrs['method']
            layout = node.op_attrs['layout']
            if size==[0,0]:
                #print('[WARNING] image.resize miss argument \'size\'')
                #exit()
                #import ipdb
                #ipdb.set_trace()
                if isinstance(self.var, np.ndarray):
                    inp_size = self.var.shape[1:3]
                else: #TF tensor
                    inp_size = [x.value for x in self.var.shape[1:3]]
                size = [inp_size[i]*(dilation_size[i]+1) for i in range(2)]
            self.var = image_resize(mode, self.var, size, method)    
        elif node.op_name=='divide':
            if node.inputs[1] in self.params.keys():
                operand_1 = self.params[node.inputs[1]].asnumpy()
                operand_1 = np.squeeze(operand_1)
            else:
                operand_1 = self.params[node.inputs[0]].asnumpy()
                operand_1 = np.squeeze(operand_1)
            if operand_1.ndim==0: # broadcast add with single value
                print('broadcast add with single value')
                offset = operand_1.tolist()
                in_channel = self.var.shape[-1].value
                operand_1 = np.array([offset for i in range(in_channel)],dtype=operand_1.dtype)
            if mode=='hw':
                operand_1, fl = self.qnn.apply(operand_1, 8)
                print(np.min(operand_1),np.max(operand_1))
                print('<<< quantize divide operand',fl,'/ 8>>>')
            self.var = divide(mode, self.var, operand_1)
        elif node.op_name=='multiply':
            if node.inputs[1] in self.params.keys():
                operand_1 = self.params[node.inputs[1]].asnumpy()
                operand_1 = np.squeeze(operand_1)
                if mode=='hw' and self.gen_fx_pb:                
                    if not isinstance(operand_1, np.ndarray):
                        tmp_np = self.sess.run(operand_1, {self.ph:self.inp_data_np})
                    else:
                        tmp_np = operand_1
                    #print(np.min(tmp_np),np.max(tmp_np),end='')
                    '''with block.suppress_stdout_stderr():
                        tmp = fixpoint(tmp_np,wordLen=16,opt=True)                    
                    operand_1 = fixpoint(tmp_np,wordLen=16,fracLen=tmp._fl)._d_fp'''
                    operand_1, fl = self.qnn.apply(tmp_np, 16)
                    print('->',np.min(operand_1),np.max(operand_1))
                    print('<<< quantize multiply param',fl,'/ 16>>>')
                    #import matplotlib.pyplot as plt
                    #plt.hist(tmp_np.flatten().tolist(), bins = 'auto')
                    #plt.hist(operand_1.flatten().tolist(), bins = 'auto')
                    #plt.show()
                    self.dump2outputdir('multiply_factor',operand_1)
            elif node.inputs[0] in self.params.keys():
                operand_1 = self.params[node.inputs[0]].asnumpy()
                operand_1 = np.squeeze(operand_1)
                if mode=='hw' and self.gen_fx_pb:                
                    if not isinstance(operand_1, np.ndarray):
                        tmp_np = self.sess.run(operand_1, {self.ph:self.inp_data_np})
                    else:
                        tmp_np = operand_1
                    #print(np.min(tmp_np),np.max(tmp_np),end='')
                    '''with block.suppress_stdout_stderr():
                        tmp = fixpoint(tmp_np,wordLen=16,opt=True)                    
                    operand_1 = fixpoint(tmp_np,wordLen=16,fracLen=tmp._fl)._d_fp'''
                    operand_1, fl = self.qnn.apply(tmp_np, 16)
                    print('->',np.min(operand_1),np.max(operand_1))
                    print('<<< quantize multiply param',fl,'/ 16>>>')
                    self.dump2outputdir('multiply_factor',operand_1)
            else:
                operand_1 = self.varDict[node.inputs[1]][mode]
                if False:#mode=='hw' and self.gen_fx_pb:                
                    if not isinstance(operand_1, np.ndarray):
                        tmp_np = self.sess.run(operand_1, {self.ph:self.inp_data_np})
                    else:
                        tmp_np = operand_1
                    print(np.min(tmp_np),np.max(tmp_np),end='')
                    '''with block.suppress_stdout_stderr():
                        tmp = fixpoint(tmp_np,wordLen=16,opt=True)                    
                    operand_1 = fixpoint(tmp_np,wordLen=16,fracLen=tmp._fl)._d_fp'''
                    operand_1, fl = self.qnn.apply(tmp_np, 16)
                    print('->',np.min(operand_1),np.max(operand_1))            
            self.var = multiply(mode, self.var, operand_1)
        elif node.op_name=='subtract':
            operand_1 = self.params[node.inputs[1]].asnumpy()
            operand_1 = np.squeeze(operand_1)
            self.var = subtract(mode, self.var, operand_1)
        elif node.op_name=='squeeze':
            axis = []#[x.value for x in node.op_attrs['axis']]
            shape = self.var.shape
            for idx,val in enumerate(shape):
                if val.value==1:
                    axis.append(idx)
            axis = tuple(axis)
            self.var = squeeze(mode, self.var, axis)
        elif node.op_name=='expand_dims':
            axis = node.op_attrs['axis']
            num_newaxis = node.op_attrs['num_newaxis']
            self.var = expand_dims(mode, self.var, axis, num_newaxis)
        elif node.op_name=='strided_slice':
            begin_id = node.op_attrs['begin'][-1].value
            end_id = node.op_attrs['end'][-1].value
            begin = [0,0,0,begin_id]
            end = [0,0,0,end_id]
            if mode=='sw':
                shape = [x.value for x in self.var.shape]
            else:
                shape = self.var.shape
            end[0:3] = shape[0:3]
            self.var = strided_slice(mode, self.var, begin, end)
        elif node.op_name=='nn.batch_flatten':
            self.var=self.var
            #self.var = tf.reshape(self.var,(1, -1))
        elif node.op_name=='nn.conv3d':
            weight = self.params[node.inputs[1]].asnumpy() # need to be '[filter_depth, filter_height, filter_width, in_channels, out_channels]'
            weight = weight.transpose(2,3,4,1,0)
            strides = [x.value for x in node.op_attrs['strides']]
            padding = node.op_attrs['padding']
            padding_sum = sum([x.value for x in padding])
            pad_mode = 'VALID' if padding_sum == 0 else 'SAME'
            node.op_attrs['padding'] = pad_mode
            node.op_attrs['groups'] = 1
            inp_shape = self.var.shape
            if inp_shape[3]==inp_shape[4] and self.layerId==0: # [N,Cin,D,H,W]
                if isinstance(self.var, np.ndarray): # need to be '[batch, in_depth, in_height, in_width, in_channels]'
                    ifm = self.var.transpose(0,2,3,4,1)
                else:
                    ifm = tf.transpose(self.var, perm=[0,2,3,4,1])
            else:
                ifm = self.var
            if mode=='hw':
                '''with block.suppress_stdout_stderr():
                    tmp = fixpoint(weight,wordLen=8,opt=True)
                    weight = fixpoint(weight, wordLen=8, fracLen=tmp._fl)._d_fp
                print('<<< quantize weight',tmp._fl,'/8>>>')'''
                weight, fl = self.qnn.apply(weight, 8)
            platform = 'onnx' if 'platform' in self.configs.keys() else None
            self.var = conv3d(mode, ifm, weight, strides, pad_mode,platform)
        elif node.op_name=='nn.max_pool3d':
            kz = [x.value for x in node.op_attrs['pool_size']]
            strides = [x.value for x in node.op_attrs['strides']]
            pad = node.op_attrs['padding']
            if sum(pad).value==0: pad_mode='VALID'
            else: pad_mode='SAME'
            platform = 'onnx' if 'platform' in self.configs.keys() else None
            self.var = max_pool3d(mode, self.var, kz, strides, pad_mode, platform)
        elif node.op_name=='nn.avg_pool3d':
            kz = [x.value for x in node.op_attrs['pool_size']]
            strides = [x.value for x in node.op_attrs['strides']]
            self.var = avg_pool3d(mode, self.var, kz, strides)
        elif node.op_name== "nn.conv3d_transpose":
            if node.inputs[1] in self.params.keys():
                weight = self.params[node.inputs[1]].asnumpy()
            else:
                weight = self.varDict[node.inputs[1]][mode]
            output_shape = node.attrs['shape'][0]
            output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[4], output_shape[1]]
            #print(output_shape)
            #import ipdb
            #ipdb.set_trace()
            self.var = conv3d_transpose(mode, self.var, weight.transpose(2,3,4,1,0),output_shape,2,'VALID')
        elif node.op_name=='yolo_reorg':
            new_shape = [x.value for x in node.op_attrs['newshape']]      
            new_shape[0]=-1
            platform = 'onnx' if 'platform' in self.configs.keys() else None
            if platform == 'onnx':
                new_shape = [new_shape[0], new_shape[2], new_shape[3], new_shape[1]]
            self.var = yolo_reorg(mode, self.var, new_shape)
        else:
            assert 0, "Unsupported Op:"+op_name
        self.varDict[node.attrs['id']][mode] = self.var
        print(self.var.shape)
        #ret = self.runTensor(self.var)
        #print(np.min(ret), np.max(ret))
        if node.attrs['id'] in [-1] and not 'compute_shape' in self.configs:
            xdf = self.sess.run(self.var,{self.ph:self.inp_data_np})
            ref = np.load('/home/tiandong/tvm/example/tests/mobilenet_v3/mobilenetv3.pytorch/0-conv.npy').transpose(0,2,3,1)
            import ipdb
            ipdb.set_trace()
            print('k')
        #print("id:",inp_node_id,'->',node.attrs['id'])
        '''if mode=='sw' and self.gen_pb and self.gen_fx_pb and node.op_name in ['multiply','divide'] and node==self.tempNodes[-1]:
            ops_cur_layer = [x.op_name for x in self.tempNodes]
            if 'clip' in ops_cur_layer:
                # indicate 'hswish' or 'hsigmoid'
                self.quantize_fm()'''
    
    def dump2outputdir(self, param_name, data):
        if self.fx_output_dir is None:
            return
        if not self.configs['dump']:
            return
        if not isinstance(data, np.ndarray) and not isinstance(data, np.float32):
            with block.suppress_stdout_stderr():
                with tf.Session() as sess:
                    data = sess.run(data, {self.ph: self.inp_data_np})
        if 'ofm' in param_name:
            print('range:',np.min(data),np.max(data))
        param_name += '_'+str(self.layerId)
        if param_name in self.output_layer_param_names[self.layerId]:
            i_tmp = 1
            while param_name+'_'+str(i_tmp) in self.output_layer_param_names[self.layerId]:
                i_tmp += 1
            name = param_name+'_'+str(i_tmp)
        else:
            name = param_name
        self.output_layer_param_names[self.layerId].append(name)
        if self.configs['dump_format']=='mat':
            scipy.io.savemat(os.path.join(self.fx_output_dir,name+'.mat'),{'value':data})
        elif self.configs['dump_format']=='npy':
            np.save(os.path.join(self.fx_output_dir,name),data)
        else:
            assert 0,'dump_format needs to be either npy or mat'
        
    def quantize_fm(self, ifm=None, fracLen=None):
        ifm_o = ifm
        if ifm is None:
            ifm = self.var
        '''if fracLen is None:
            if self.layerId in self.ctpDict.keys():
                fracLen = self.ctpDict[self.layerId]
            else:
                with block.suppress_stdout_stderr():
                    with tf.Session() as sess:
                        sw = sess.run(ifm, {self.ph: self.inp_data_np})
                    tmp = fixpoint(sw, wordLen=self.fm_wordLen, opt=True, dtype=self.dtype)
                fracLen = tmp._fl
        print('<<< feature map quantization',fracLen,'/',self.fm_wordLen,'>>>')
        self.layer_fracLen_dict['FM'][self.layerId] = fracLen  
        self.var = tf_float2fx_round(ifm, self.fm_wordLen, fracLen)
        self.dump2outputdir('hw_ofm',self.var)
        return self.var'''        
        if fracLen is None:
            if self.layerId in self.ctpDict.keys():
                fracLen = self.ctpDict[self.layerId]
            else:
                with block.suppress_stdout_stderr():
                    with tf.Session() as sess:
                        sw = sess.run(ifm, {self.ph: self.inp_data_np})
                fracLen = self.qnn.search(sw, self.fm_wordLen)
        print('<<< quantize feature map',fracLen,'/',self.fm_wordLen,'>>>')
        self.layer_fracLen_dict['FM'][self.layerId] = fracLen 
        self.var = self.qnn.convert(ifm, self.fm_wordLen, fracLen, symbolic=True)
        self.dump2outputdir('fm',self.var)
        return self.var
    
    def compute_error(self, inp1, inp2):
        assert inp1.shape==inp2.shape, "Shape mismatch when computing error"
        dif = abs(inp1-inp2)
        dif = np.reshape(dif,(dif.size))
        err = np.sum(dif)/dif.size
        return err
    
    def runTensor(self, tensor, inp=None):
        with block.suppress_stdout_stderr():
            with tf.Session() as ses:
                if inp is None or not self.gen_pb:
                    ret = ses.run(tensor)
                else:
                    ret = ses.run(tensor, feed_dict={self.inp:inp})
            return ret 
            
    def saveRawInput(self, savePath, inp):
        fw = open(savePath, 'w')
        assert len(inp.shape)==3 # h,w,c
        i_h,i_w,i_c = inp.shape
        inp = inp.astype(np.int8)
        for i in range(i_h):
            for j in range(i_w):
                for c in range(i_c):
                    print(inp[i][j][c],file=fw,end=' ')
                print(file=fw)
        fw.close()
    
    def getStride(self, inpz,outz,kz):
        s = 1
        if (inpz+s-1)//(outz+kz-1)<1: return 1
        while not s==round((inpz+s-1)/(outz+kz-1)):
            s+=1
        return s
