import numpy as np
import copy
import os
from collections import defaultdict
from collections import OrderedDict
import tvm
import json
import math
class outWriter:
    def __init__(self,layers,shapeDict,params):
        self.layers = layers
        self.params = params
        self.shape_dict = shapeDict # tvm relay ir node_id -> shape(4-d tuple)
        self.main_ops = ['nn.conv2d','nn.dense','nn.conv3d','nn.conv2d_transpose','nn.conv3d_transpose']
        self.activation_ops = ['nn.relu','clip','tanh','nn.leaky_relu','sigmoid']
        self.pool_ops = ['nn.avg_pool2d','nn.max_pool2d','nn.max_pool3d','nn.global_avg_pool2d']
        self.template = {
                        'type':None,
                        'input':[],
                        'output':[],
                        'input_size':None,
                        'output_size':None,
                        'kernel_size':None,
                        'padding_size':None,
                        'activation':None,
                        'pool':None,
                        'pool_size':None,
                        'pool_stride':None,
                        'pool_padding':None,
                        'groups':None,
                        'shuffle_groups':None,
                        'extra_concat_inputs':[],
                        'concat_after_conv':False,
                        'residual_input_layer_id':None,
                        'ops_after_concat':[],
                        'upsampling_scale':[]
                        }
        self.id_output_map = defaultdict(list)
        self.id_input_map = {}
        self.layer_infos = []
        self.util_dict = {}
        self.layer_concat_node_dict = {}
        self.first_layer_data_format = None
        
    def parse(self):
        # schedule blocks as ASAP but still obey topological order
        '''tmp_dict = defaultdict(list)
        for item in self.layers:
            if len(item[0].inputs)>0:
                main_inp = max(item[0].inputs) #item[0].inputs[0]
            else:
                main_inp = -1
            tmp_dict[main_inp].append(item)
        self.layers = []
        for item in sorted(tmp_dict.keys()):
            for u in tmp_dict[item]:
                self.layers.append(u)'''
        # parse
        ids = []
        for item in self.layers:
            ids.append([x.attrs['id'] for x in item])
        # type(conv/fc/deconv),size,kernel_size,padding_size
        for nodes in self.layers:
            layer_info = copy.deepcopy(self.template)
            # activation
            activation_nodes = [x for x in nodes if x.op_name in self.activation_ops]
            if len(activation_nodes)>0:
                activation_type = activation_nodes[0].op_name
                if activation_type == 'clip':
                    div_nodes = [x for x in nodes if x.op_name=='divide']
                    if len(div_nodes)>0:
                        cur_ids = [x.attrs['id'] for x in nodes]
                        '''div_inp_node_id = [x for x in div_nodes[0].inputs if x in cur_ids][0]
                        div_inp_node = [x for x in nodes if x.attrs['id']==div_inp_node_id][0]
                        div_inp_op = div_inp_node.op_name
                        if div_inp_op == 'multiply':
                            activation_type = 'h-swish'
                        else:
                            activation_type = 'h-sigmoid' '''
                        div_out_node = [x for x in nodes if div_nodes[0].attrs['id'] in x.inputs]
                        if len(div_out_node)>0 and div_out_node[0].op_name == 'multiply':
                            activation_type='h-swish'
                        else:
                            activation_type='h-sigmoid'
                layer_info['activation'] = activation_type
            # pooling
            pool_nodes = [x for x in nodes if x.op_name in self.pool_ops]
            if len(pool_nodes)>0:
                pool_type = pool_nodes[0].op_name
                layer_info['pool'] = pool_type
                if pool_type == 'nn.global_avg_pool2d':
                    layer_info['pool'] = 'nn.avg_pool2d'
                    inp_id = pool_nodes[0].inputs[0]
                    in_shape = self.shape_dict[inp_id]
                    size = in_shape[1].value
                    layer_info['pool_stride'] = [tvm.expr.IntImm('int',1),tvm.expr.IntImm('int',1)]
                    layer_info['pool_size'] = [tvm.expr.IntImm('int',size), tvm.expr.IntImm('int',size)]
                    layer_info['pool_padding'] = [tvm.expr.IntImm('int',0),tvm.expr.IntImm('int',0),tvm.expr.IntImm('int',0),tvm.expr.IntImm('int',0)]
                else:
                    pool_stride = pool_nodes[0].op_attrs['strides']
                    layer_info['pool_stride'] = pool_stride
                    pool_size = pool_nodes[0].op_attrs['pool_size']
                    layer_info['pool_size'] = pool_size
                    pool_padding = pool_nodes[0].op_attrs['padding']
                    layer_info['pool_padding'] = pool_padding
            # padding_size
            pad_nodes = [x for x in nodes if x.op_name=='nn.pad']
            if len(pad_nodes)>0:
                pad_width = pad_nodes[0].op_attrs['pad_width']
                tmp = []
                for u in pad_width:
                    tmp.append([x.value for x in u])
                if len(tmp)==5:
                    tmp = [tmp[0],tmp[2],tmp[3],tmp[4]]
                layer_info['padding_size'] = tmp
            # type
            main_nodes = [x for x in nodes if x.op_name in self.main_ops]
            if len(main_nodes)>0:
                op = main_nodes[0].op_name
                layer_info['type'] = op
                # kernel_size
                kernel_id = main_nodes[0].inputs[1]
                kernel_shape = self.shape_dict[kernel_id]
                if 'kernel_layout' in main_nodes[0].op_attrs.keys():
                    format = main_nodes[0].op_attrs['kernel_layout']
                    if format=='OIHW': # -> HWIO
                        kernel_shape = [kernel_shape[2], kernel_shape[3], kernel_shape[1], kernel_shape[0]]
                layer_info['kernel_size'] = kernel_shape          
                if op=='nn.conv2d' or op=='nn.conv3d':
                    layer_info['groups'] = main_nodes[0].op_attrs['groups']
                    if len(self.layer_infos)==0 and 'data_layout' in main_nodes[0].op_attrs.keys():
                        self.first_layer_data_format = main_nodes[0].op_attrs['data_layout']
                    # channel shuffle
                    ops = [x.op_name for x in nodes]
                    if 'reshape' in ops and 'transpose' in ops:
                        reshape_nodes = [x for x in nodes if x.op_name=='reshape']
                        inp_shape = self.shape_dict[reshape_nodes[0].inputs[0]]
                        reshape_shape = [x.value for x in reshape_nodes[0].op_attrs['newshape']]
                        if not isinstance(inp_shape[0], int):
                            in_channel = inp_shape[3].value # assume NHWC
                        else:
                            in_channel = inp_shape[3]
                        groups, num_per_group = reshape_shape[1], reshape_shape[2] # since we have transpose
                        assert groups*num_per_group==in_channel, "groups {}, num_per_group {} cannot match in_channel {}".format(groups, num_per_group, in_channel)
                        layer_info['shuffle_groups'] = groups
                    # padding 
                    padding = main_nodes[0].op_attrs['padding']
                    if padding=='SAME':
                        if op=='nn.conv3d':
                            tmp = [[0,0],[kernel_shape[2]//2,kernel_shape[2]//2],[kernel_shape[3]//2,kernel_shape[3]//2], [kernel_shape[4]//2, kernel_shape[4]//2]]
                        else:
                            tmp = [[0,0],[kernel_shape[0]//2,kernel_shape[0]//2], [kernel_shape[1]//2, kernel_shape[1]//2],[0,0]]
                        layer_info['padding_size'] = tmp
            else: #if len(nodes)<=2:
                ops = [x.op_name for x in nodes]
                if 'nn.avg_pool2d' in ops or 'nn.avg_pool3d' in ops or 'nn.max_pool2d' in ops or 'nn.global_avg_pool2d' in ops or 'nn.max_pool3d' in ops :
                    layer_info['type'] = 'single_pool'
                elif 'multiply' in ops:
                    layer_info['type'] = 'scale_by_fm'
            # concatenate
            concat_nodes = [x for x in nodes if x.op_name=='concatenate']
            if len(concat_nodes)>0:
                concat_node = concat_nodes[0]
                self.layer_concat_node_dict[len(self.layer_infos)] = concat_node
                concat_inps = []
                for item in concat_node.inputs:
                    for lid in range(len(self.layers)):
                        if item in ids[lid]:
                            #if not lid==len(self.layer_infos) and not lid+1==len(self.layer_infos):
                            concat_inps.append(lid)
                            break
                if len(self.layer_infos) in concat_inps:
                    concat_inps.remove(len(self.layer_infos))
                elif len(self.layer_infos)-1 in concat_inps:
                    concat_inps.remove(len(self.layer_infos)-1)
                layer_info['extra_concat_inputs'] = concat_inps
                if len(main_nodes)>0 and concat_node.attrs['id']>main_nodes[0].attrs['id'] and len(concat_inps)>0:
                    layer_info['concat_after_conv'] = True
                    opnodes_after_concat = [x for x in nodes if x.attrs['id']>concat_node.attrs['id']]
                    self.util_dict[len(self.layer_infos)] = opnodes_after_concat
                    ops_after_concat = [x.op_name for x in opnodes_after_concat]
                    layer_info['ops_after_concat'] = ops_after_concat
                elif layer_info['type'] == 'single_pool' and len(concat_inps)>0:
                    layer_info['concat_after_conv'] = True
                    opnodes_after_concat = [x for x in nodes if x.attrs['id']>concat_node.attrs['id']]
                    self.util_dict[len(self.layer_infos)] = opnodes_after_concat
                    ops_after_concat = [x.op_name for x in opnodes_after_concat]
                    layer_info['ops_after_concat'] = ops_after_concat
            # residual input layer id
            add_nodes = [x for x in nodes if x.op_name=='add']
            cur_layer_ids = ids[len(self.layer_infos)]
            residual_node_input_id = []
            tmp = []
            for item in ids:
                for x in item:
                    tmp.append(x)
            for item in add_nodes:
                for u in item.inputs:
                    if not u in cur_layer_ids and u in tmp:
                        residual_node_input_id.append(u)
            if len(residual_node_input_id)>0:
                for ii, node_ids in enumerate(ids):
                    if residual_node_input_id[0] in node_ids:
                        layer_info['residual_input_layer_id'] = ii
                        break
            if len(add_nodes)>0:
                layer_info['has_bias'] = True
            else:
                layer_info['has_bias'] = False
            # input_size, output_size
            fms = [x for x in nodes if 'op_name' in dir(x)]
            input_size = self.shape_dict[fms[0].inputs[0]]
            if not isinstance(input_size[0], int):
                input_size = [x.value for x in input_size]
            layer_info['input_size'] = tuple(input_size[1:])
            output_size = self.shape_dict[fms[-1].attrs['id']]
            if not isinstance(output_size[0], int):
                output_size = [x.value for x in output_size]
            layer_info['output_size'] = tuple(output_size[1:])
            # input
            inp = []
            for item in fms:
                for x in item.inputs:
                    if x not in self.params.keys():
                        inp.append(x)
            inp = set(inp)
            cur = [x.attrs['id'] for x in fms]
            inp_id = [x for x in inp if x not in cur]#fms[0].inputs
            self.id_input_map[len(self.layer_infos)] = inp_id
            for i,item in enumerate(ids):
                for x in inp_id:
                    if x in item:
                        layer_info['input'].append(i)
                        self.id_output_map[i].append(len(self.layer_infos)) 
            # upsampling_scale
            us_nodes = [x for x in nodes if x.op_name=='nn.upsampling']
            if len(us_nodes)==0:
                layer_info['upsampling_scale'] = 0
            else:
                scale = us_nodes[0].op_attrs['scale']
                layer_info['upsampling_scale'] = scale
            # save layer info
            self.layer_infos.append(layer_info)
        # output
        for i, layer_info in enumerate(self.layer_infos):
            self.layer_infos[i]['output'] = self.id_output_map[i]
            # shortcut info
            id = i
            ''' find current layer's nodes that output to other layers '''
            cur_ids = [x.attrs['id'] for x in self.layers[id]]
            olayer_curnode_map = {} # output layer id <- cur_layer node id
            for layer_id in self.id_output_map[id]:
                for iid in self.id_input_map[layer_id]:
                    if iid in cur_ids:
                        olayer_curnode_map[layer_id] = iid
            activation_nodes = [x for x in self.layers[id] if x.op_name in self.activation_ops]
            if len(activation_nodes)>0:
                act_id = activation_nodes[0].attrs['id']
            else:
                act_id = float('inf')
            pool_nodes = [x for x in self.layers[id] if x.op_name in self.pool_ops]
            if len(pool_nodes)>0:
                pool_id = pool_nodes[0].attrs['id']
            else:
                pool_id = float('inf')
            node_sorted = sorted(olayer_curnode_map.items(), key=lambda kv: kv[1])
            
            if len(node_sorted)>1:
                normal_onode_id = node_sorted[0][1]
                # "output_choice": [] //0->nothing; 1->pooling; 2->relu; 3 relu+pooling; [normal, shortcut]
                as_normal = self.encode_output_choice(normal_onode_id, act_id, pool_id)
            else:
                as_normal = 3
            if len(node_sorted)==0:
                as_shortcut = -1
            elif len(node_sorted)==1: # if cur_layer is not the input of sequential normal layer, then should be counted as a shortcut output to further layers
                if id<len(self.layer_infos)-1 and id not in self.layer_infos[id+1]['input']:
                    sc_onode_id = node_sorted[0][1]
                    as_shortcut = self.encode_output_choice(sc_onode_id, act_id, pool_id)
                else:
                    as_shortcut = -1
            else:
                sc_onode_id = node_sorted[1][1]
                as_shortcut = self.encode_output_choice(sc_onode_id, act_id, pool_id)
            self.layer_infos[i]['output_choice'] = [as_normal, as_shortcut]
            # residual pos in current layer//0-> before Relu, 1-> before pooling, 2->after all, 3 ->no residual
            inode_ids = self.id_input_map[i]  
            if self.layer_infos[i]['residual_input_layer_id'] is not None and len(inode_ids)>1:
                cur_nodes = []
                for j,node in enumerate(self.layers[i]):
                    for iid in inode_ids:
                        '''if isinstance(node, tuple):
                            j = 0
                            while not isinstance(node[j], tvm.relay.backend.graph_runtime_codegen.OpNode):
                                j+=1
                            node = node[j]'''
                        if not j==0 and iid in node.inputs:
                            cur_nodes.append(node.attrs['id'])
                tmp = set([self.encode_residual_pos(x, act_id, pool_id) for x in cur_nodes])
                if len(tmp)==0:
                    in_shortcut_pos = 3
                else:
                    in_shortcut_pos = list(tmp)[0]
            else:
                in_shortcut_pos = 3
            self.layer_infos[i]['residual_pos'] = in_shortcut_pos
            #import ipdb
            #ipdb.set_trace()
            
    def encode_output_choice(self, x, act_id, pool_id):
        if x>=pool_id and x>=act_id:
            pos = 3
        elif x<pool_id and x>=act_id:
            pos = 2
        elif x>=pool_id and x<act_id:
            pos = 1
        else:
            pos = 0
        return pos
    
    def encode_residual_pos(self, x, act_id, pool_id):
        if pool_id==float('inf') and act_id==float('inf'):
            return 2
        if x<pool_id and x<act_id:
            pos = 0
        elif x<pool_id and x>=act_id:
            pos = 1
        elif x>=pool_id and x<act_id:
            pos = 2
        else:
            pos = 3
        return pos
    
    def get_mac_number(self, ir): # fused weights
        mac_tot = 0
        param_tot = 0
        mac_conv = 0
        mac_conv_group = 0
        mac_se = 0
        print('='*5,'Mul+Add','='*5)
        for i in range(ir['num'][0]):
            type = ir['type'][i]
            input_size = ir['input_size'][i]
            output_size = ir['output_size'][i]
            padding_size = ir['pre_padding_size'][i]
            ker_stride = ir['ker_stride'][i]
            ker_size = ir['ker_size'][i]
            group = ir['group'][i]
            residual = ir['residual'][i]
            activation = ir['activation_type'][i]
            print('type:',type)
            if type == 1: #conv2d
                D,H,W,C = input_size
                Cout = output_size[-1]
                dpad, hpad, wpad = padding_size
                D += dpad*2
                H += hpad*2
                W += wpad*2
                Kd, Kh, Kw = ker_size
                Sd, Sh, Sw = ker_stride
                #print('\ninput:',input_size,'out_nopool:',[((D-Kd)/Sd+1),((W-Kw)/Sw+1),((H-Kh)/Sh+1),Cout],'pad:',padding_size,'stride:',ker_stride,'kz:',ker_size)
                Wo = math.floor((W-Kw)/Sw+1)
                Ho = math.floor((H-Kh)/Sh+1)
                Do = math.floor((D-Kd)/Sd+1)
                print('\ninput:',input_size,'out_nopool:',[Do, Wo, Ho ,Cout],'pad:',padding_size,'stride:',ker_stride,'kz:',ker_size)
                if group == 1:
                    #mac = Cout*(W-Kw+1)*(H-Kh+1)/(Sh*Sw)*(2*Kh*Kw*C)
                    mac = Cout*Wo*Ho*Do*(2*Kh*Kw*Kd*C-1)                    
                    mac = Cout*Wo*Ho*Do*(Kh*Kw*Kd*C) # mul only                 
                    param = Cout*C*Kh*Kw*Kd #+Cout
                    '''if not self.layer_infos[i]['has_bias']:
                        mac -= Cout*(W-Kw+1)*(H-Kh+1)/(Sh*Sw)
                        param -= Cout'''
                    mac_conv += mac
                else: # group conv
                    mac = Cout/group*Wo*Ho*Do*((2*Kh*Kw*Kd-1)*(C-1))
                    mac = Cout/group*Wo*Ho*Do*(Kh*Kw*Kd*C) # mul only
                    param = Cout/group*Kh*Kw*Kd*C #+Cout
                    mac_conv_group += mac
                    '''if not self.layer_infos[i]['has_bias']:
                        mac -= Cout/group*(W-Kw+1)*(H-Kh+1)/(Sh*Sw)
                        param -= Cout
                        mac_conv_group -= Cout/group*(W-Kw+1)*(H-Kh+1)/(Sh*Sw)'''
            elif type == 4: # scale in MobilenetV3
                D,H,W,C = input_size
                mac = D*H*W*C
                param = 0
            elif type == 0:
                Cin, Cout = input_size[-1], output_size[-1]
                mac = (2*Cin-1)*Cout
                param = Cin*Cout #+Cout
            else:
                mac = 0
                param = 0
            if residual == 1:
                D,H,W,C = output_size
                #mac += H*W*C*D
            if activation==3: #h-swish
                D,H,W,C = input_size
                #mac += D*H*W*C*2
            elif activation==4: #h-sigmoid
                D,H,W,C = input_size
                #mac += D*H*W*C*3
            print('#',i+1,'Mul+Add =',mac,'(param:',param,')')
            mac_tot += mac
            param_tot += param
            if (type==0 or type==4) and i not in [ir['num'][0]-1, ir['num'][0]-2]:
                mac_se += mac;
        print('TOTAL #(MUL+ADD) = ',mac_tot,'(',mac_tot/(10**6),'M)','(',mac_tot/(10**9),'G)','(',mac_tot/(10**12),'B)')
        print('Param =',param_tot,'(',param_tot/(10**6),'M)')
        print('conv:',mac_conv/mac_tot,'conv_group:',mac_conv_group/mac_tot)
        print(mac_se, mac_se/mac_tot)
    
    def get_mac_number_3d(self, ir): # fused weights
        mac_tot = 0
        param_tot = 0
        mac_conv = 0
        mac_conv_group = 0
        print('='*5,'Mul+Add','='*5)
        for i in range(ir['num'][0]):
            type = ir['type'][i]
            input_size = ir['input_size'][i]
            #if i==0: input_size = [input_size[3],input_size[0],input_size[1],input_size[2]]
            output_size = ir['output_size'][i]
            padding_size = ir['pre_padding_size'][i]
            ker_stride = ir['ker_stride'][i]
            ker_size = ir['ker_size'][i]
            group = ir['group'][i]
            residual = ir['residual'][i]
            if type == 1: #conv3d
                D,H,W,C = input_size
                dpad,hpad,wpad = padding_size
                H += hpad*2
                W += wpad*2
                D += dpad*2
                Do,Ho,Wo,Co = output_size
                Kd, Kh, Kw = ker_size
                Sd, Sh, Sw = ker_stride
                mac = Co*(C*(W-Kw+1)*(H-Kh+1)*(D-Kd+1)/(Sd*Sh*Sw)*(2*Kd*Kh*Kw-1))
                mac = Co*((W-Kw)/Sw+1)*((H-Kh)/Sh+1)*((D-Kd)/Sd+1)*((2*Kd*Kh*Kw-1)*C+C-1)
                #print('\ninput:',input_size,'out_nopool:',[((D-Kd)/Sd+1),((W-Kw)/Sw+1),((H-Kh)/Sh+1)],'pad:',padding_size,'stride:',ker_stride,'kz:',ker_size)
                param = Co*C*Kh*Kw+Co
                mac_conv += mac
            elif type == 0:
                Cin, Cout = input_size[-1], output_size[-1]
                mac = (2*Cin)*Cout
                mac = (2*Cin-1)*Cout
                param = Cin*Cout+Cout
            else:
                mac = 0
                param = 0
            if residual == 1:
                Do,Ho,Wo,Co = output_size
                mac += Co*Do*Ho*Wo
            #print('#',i+1,'Mul+Add =',mac,'(param:',param,')')
            mac_tot += mac
            param_tot += param
        print('TOTAL #(MUL+ADD) = ',mac_tot,'(',mac_tot/(10**6),'M)','(',mac_tot/(10**9),'G)','(',mac_tot/(10**12),'B)')
        #print('Param =',param_tot,'(',param_tot/(10**6),'M)')
        #print('conv:',mac_conv/mac_tot,'conv_group:',mac_conv_group/mac_tot)
    
    
    def setWriteDir(self,dir):
        self.out_dir = dir
        
    def write2file(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        with open(os.path.join(self.out_dir,'RAW_IR.txt'),'w') as f:
            for i,item in enumerate(self.layer_infos):
                print('='*20, file=f)
                print('id :',i, file=f)
                for key in sorted(item.keys()):
                    print(key,':',item[key], file=f)
    
    def write2terminal(self):
        for i,item in enumerate(self.layer_infos):
            print('='*10,i,'='*10)
            print(item)
    
    def write2file_opu_ir(self):
        # translate layer_infos
        ir = OrderedDict()
        ir['num'] = [len(self.layer_infos)]
        ir['type'] = []
        ir['input_ddr_addr'] = []
        ir['output_ddr_addr'] = []
        ir['input_size'] = []
        ir['output_size'] = []
        ir['ker_size'] = []
        ir['ker_stride'] = []
        ir['pooling'] = []
        ir['pooling_size'] = []
        ir['pooling_stride'] = []
        ir['pooling_type'] = []
        ir['pre_padding'] = []
        ir['pre_padding_size'] = []
        #ir['post_padding'] = [0 for i in range(len(self.layer_infos))]
        #ir['post_padding_size'] = [[0,0,0] for i in range(len(self.layer_infos))]
        ir['activation_type'] = []
        ir['residual'] = []
        ir['shortcut_source'] = []
        ir['output_choice'] = []
        ir['res_position'] = []
        ir['pool_pad'] = []
        ir['pool_pad_size'] = []
        ir['group'] = [] #group convolution 分组数量，depthwise的这个值等于输入channel数量]
        ir['channel_shuffle'] = []#该层的输出是否需要做shuffle， 如果需要则放shuffle完成后group的数量，没有shuffle放1]
        ir['upsampling_scale'] = []
        for i in range(len(self.layer_infos)):
            layer = self.layer_infos[i]
            # 'type'
            if layer['type']=='nn.conv2d' or layer['type']=='nn.conv3d':
                ltype = 1
            elif layer['type']=='single_pool':
                ltype = 3
            elif layer['type']=='scale_by_fm':
                ltype = 4
            elif layer['type']=='nn.conv2d_transpose' or layer['type']=='nn.conv3d_transpose':
                ltype = 2
            else:
                ltype = 0
            ir['type'].append(ltype)
            # 'input_ddr_addr'
            input = [x+1 for x in layer['input']]
            if len(input)==0:
                input=[0]
                input_except_residual = input
            elif len(input)==1:
                input_except_residual = input
            else: # remove residual input from input_ddr_addr(normal+concat)
                input_except_residual = [] 
                extra_concat_input_ids = [x+1 for x in layer['extra_concat_inputs']]
                if layer['residual_input_layer_id'] is None:
                    tmp = -2
                else:
                    tmp = layer['residual_input_layer_id']
                for ii in range(len(input)):
                    if input[ii] in extra_concat_input_ids or not input[ii]==tmp+1:#input[ii]==i: #assume first input is the main input to conv2d
                        input_except_residual.append(input[ii])
            ir['input_ddr_addr'].append(input_except_residual)
            #'output_ddr_addr':[],
            output = [x+1 for x in layer['output']]
            if len(output)==0:
                output=[-1]
            ir['output_ddr_addr'].append(output)
            #'input_size':[],
            inpz = [x for x in layer['input_size']]
            while len(inpz)<=3:
                inpz.insert(0,1)
            if layer['padding_size'] is None:
                pad = [0,0,0]
            else:
                pad = [sum(x) for x in layer['padding_size']][1:]
                if pad[0]==0 and (not pad[1]==0 or not pad[2]==0):
                    pad = [pad[1], pad[2], pad[0]]
            #inpz = [pad[ii]+inpz[ii] for ii in range(3)]
            for ii in range(3):
                if not isinstance(inpz[ii], int):
                    inpz[ii] = inpz[ii].value
            '''if i==0:
                if self.first_layer_data_format=='NCHW':
                    inpz = [inpz[1], inpz[2], inpz[0]]'''
            while len(inpz)<4:
                inpz.insert(0,1)
            ir['input_size'].append(inpz)
            #'output_size':[],
            outz = [x for x in layer['output_size']]
            while len(outz)<4:
                outz.insert(0,1)
            ir['output_size'].append(outz)
            #'ker_size':[],
            if layer['kernel_size'] is None:
                kz = [1,1,1]
            elif len(layer['kernel_size'])==4:
                kz = [layer['kernel_size'][0],layer['kernel_size'][1],1]
            elif len(layer['kernel_size'])==5:
                kz = list(layer['kernel_size'][2:])
            if layer['type']=='nn.dense':
                kz = [1,1,1]
            if layer['type']=='nn.conv2d':
                kz = [kz[2],kz[0],kz[1]]
            ir['ker_size'].append(kz)
            #'pooling':[], 
            #'pooling_size':[],
            #'pooling_stride':[],
            #'pooling_type':[],
            if layer['pool'] is None:
                pool = 0
                size = [1,1,1]
                stride = [1,1,1]
                type = 0
            else:
                pool = 1
                size = [x.value for x in layer['pool_size']]
                if len(size)<3:
                    size.insert(0,1) #size.append(1)
                stride = [x.value for x in layer['pool_stride']]
                if len(stride)<3:
                    stride.insert(0,1) #stride.append(1)
                if layer['pool']=='nn.max_pool2d' or layer['pool']=='nn.max_pool3d':
                    type = 1
                elif layer['pool']=='nn.avg_pool2d' or layer['pool']=='nn.global_avg_pool2d':
                    type = 2
            ir['pooling'].append(pool)
            ir['pooling_size'].append(size)
            ir['pooling_stride'].append(stride)
            ir['pooling_type'].append(type)
            #'ker_stride':[],
            main_node = [x for x in self.layers[i] if x.op_name in self.main_ops]
            if len(main_node)>0:
                node = main_node[0]
                if 'strides' in node.op_attrs.keys():
                    s = node.op_attrs['strides']
                else:
                    id = node.attrs['id']
                    outz = [x.value for x in self.shape_dict[id]][1:]
                    for ii in range(len(kz)):
                        if not isinstance(kz[ii], int):
                            kz[ii] = kz[ii].value
                    s = 1
                    if round((inpz[1]+s-1)/(outz[0]+kz[1]-1))>1:
                        while not s==round((inpz[1]+s-1)/(outz[0]+kz[0]-1)):
                            s+=1
            else:
                s = 1
            if isinstance(s, int):
                ir['ker_stride'].append([1,s,s])
            else:
                ir['ker_stride'].append([x.value for x in list(s)])
            #'pre_padding':[],
            #'pre_padding_size':[],
            #ir['pre_padding'].append(0)
            #ir['pre_padding_size'].append([0,0,0])
            #'post_padding':[],
            #'post_padding_size':[],
            if layer['padding_size'] is not None:
                tmp = [x[0] for x in layer['padding_size']][1:]
                if not sum(tmp)==0:
                    if layer['type']=='nn.conv2d':
                        tmp = [tmp[2],tmp[0],tmp[1]]
                    ir['pre_padding_size'].append(tmp)
                    ir['pre_padding'].append(1)
                else:    
                    ir['pre_padding_size'].append([0,0,0])
                    ir['pre_padding'].append(0)
            else:    
                ir['pre_padding_size'].append([0,0,0])
                ir['pre_padding'].append(0)
            #'activation_type':[],
            activation = layer['activation']
            act_type = {'nn.relu':1,'clip':1,'nn.leaky_relu':2,'h-sigmoid':4,'h-swish':3}
            if activation in act_type.keys():
                ir['activation_type'].append(act_type[activation])
            else:
                ir['activation_type'].append(0)
            #'residual':[],
            #'shortcut_source':[],
            if len(input)>1:
                extra_concat_input_ids = [x+1 for x in layer['extra_concat_inputs']]
                extra_inputs = [x for x in input if x not in extra_concat_input_ids]
                #residual_inputs = [x for x in extra_inputs if not x==i+1-1] # +1:layerId from 1-inf, -1:previous layer
                #residual_inputs = extra_inputs[1:]
                if layer['residual_input_layer_id'] is None:
                    tmp = -2
                else:
                    tmp = layer['residual_input_layer_id']
                residual_inputs = [x for x in extra_inputs if x==tmp+1]
                ir['shortcut_source'].append(residual_inputs)
                if len(residual_inputs)>0:
                    ir['residual'].append(1)
                else:
                    ir['residual'].append(0)
            else:
                ir['residual'].append(0)
                ir['shortcut_source'].append([])
            #'output_choice':[],
            #'res_position':[],
            ir['output_choice'].append(layer['output_choice'])
            ir['res_position'].append(layer['residual_pos'])
            #'pool_pad':[],
            #'pool_pad_size':[],
            pool_pad = layer['pool_padding']
            if pool_pad is None:
                ir['pool_pad'].append(0)
                pool_pad = [0,0,0,0,0]
            elif sum(pool_pad).value==0:
                ir['pool_pad'].append(0)
                #pool_pad = [x.value for x in pool_pad]
                #pool_pad.append(0)
                pool_pad = [0,0,0,0,0]
            else:
                ir['pool_pad'].append(1)
                pool_pad = [x.value for x in pool_pad]
                if len(pool_pad)==2:
                    pool_pad = [pool_pad[0],pool_pad[1],pool_pad[0],pool_pad[1]]
                pool_pad.append(0)
            ir['pool_pad_size'].append(pool_pad)
            #'group':[],#group convolution 分组数量，depthwise的这个值等于输入channel数量]
            if layer['type'] is None:
                ir['group'].append(1)
            else:
                if layer['groups'] is None:
                    layer['groups'] = 1
                ir['group'].append(layer['groups'])
            #'channel_shuffle':[]#该层的输出是否需要做shuffle， 如果需要则放shuffle完成后group的数量，没有shuffle放1]
            if layer['shuffle_groups'] is not None:
                ir['channel_shuffle'].append(layer['shuffle_groups'])
            else:
                ir['channel_shuffle'].append(1) 
            # fix_point
            # TODO
            # upsampling_scale
            ir['upsampling_scale'].append(layer['upsampling_scale'])
        # deal with special case with 'concatenate after conv' in densenet, modify input_ddr_addr  
        # propagate ops after concatenate to predecessors, starting from the destination node
        concat_after_conv = False
        for i in range(ir['num'][0]-1, -1,-1):
            if self.layer_infos[i]['concat_after_conv'] and len(self.layer_infos[i]['extra_concat_inputs'])>0:
                print('#'*10,i, self.layer_infos[i]['concat_after_conv'],self.layer_infos[i]['ops_after_concat'])
                #import ipdb
                #ipdb.set_trace()
                concat_after_conv = True
                concat_node = self.layer_concat_node_dict[i]
                cur_ids = [x.attrs['id'] for x in self.layers[i]]
                concat_cur_layer_inp = [x for x in concat_node.inputs if x in cur_ids] # should be only one as there's "concat after conv" in cur layer
                #ir['input_size'][i+1][-1] = ir['output_size'][i][-1] # input channel of next layer = output channel number of concat
                ir['output_size'][i][-1] = self.shape_dict[concat_cur_layer_inp[0]][-1].value # output channel number of cur layer = cur layer's input to concat
                extra_concat_input_ids = [x+1 for x in self.layer_infos[i]['extra_concat_inputs']]
                print([x-1 for x in extra_concat_input_ids])
                for item in extra_concat_input_ids:  
                    #if item in ir['input_ddr_addr'][i]:
                    #    ir['input_ddr_addr'][i].remove(item)
                    #    print('layer',i,'remove input',item-1)
                    #for ii in ir['output_ddr_addr'][i]:
                    #    if item not in ir['input_ddr_addr'][ii-1]:
                    #        ir['input_ddr_addr'][ii-1].append(item)
                    #    print('layer',ii-1,'add input',item-1)
                    ops_after_concat = self.layer_infos[i]['ops_after_concat']
                    if len(ops_after_concat)>0:
                        if len([x for x in ops_after_concat if x in self.pool_ops])>0:
                            ir['pooling'][item-1] = ir['pooling'][i]
                            ir['pooling_size'][item-1] = ir['pooling_size'][i]
                            ir['pooling_stride'][item-1] = ir['pooling_stride'][i]
                            ir['pooling_type'][item-1] = ir['pooling_type'][i]
                            ir['pool_pad'][item-1] = ir['pool_pad'][i]
                            ir['pool_pad_size'][item-1] = ir['pool_pad_size'][i] 
                            # update shape !!
                            ir['output_size'][item-1][1:3] = ir['output_size'][i][1:3]
                            print('pool ->',item-1)
                        if len([x for x in ops_after_concat if x in self.activation_ops])>0:
                            ir['activation_type'][item-1] = ir['activation_type'][i]
                            print('activation ->',item-1)
                        if self.layer_infos[item-1]['concat_after_conv']:
                            for cop in ops_after_concat:
                                if cop not in self.layer_infos[item-1]['ops_after_concat']:
                                    self.layer_infos[item-1]['ops_after_concat'].append(cop)
        # propagate input/output change, starting from the root(otherwise incorrect input would be propagated to all predecessors)
        concat_after_conv = False
        for i in range(ir['num'][0]):
            if self.layer_infos[i]['concat_after_conv'] and len(self.layer_infos[i]['extra_concat_inputs'])>0:
                print('#'*10,i, self.layer_infos[i]['concat_after_conv'],self.layer_infos[i]['ops_after_concat'])
                concat_after_conv = True
                extra_concat_input_ids = [x+1 for x in self.layer_infos[i]['extra_concat_inputs']]
                print([x-1 for x in extra_concat_input_ids])
                for item in extra_concat_input_ids:  
                    if item in ir['input_ddr_addr'][i]:
                        ir['input_ddr_addr'][i].remove(item)
                        print('layer',i,'remove input',item-1)
                    for ii in ir['output_ddr_addr'][i]:
                        if item not in ir['input_ddr_addr'][ii-1]:
                            ir['input_ddr_addr'][ii-1].append(item)
                        print('layer',ii-1,'add input',item-1)
        
            
        # update output
        if concat_after_conv:
            tmp = [[] for i in range(len(ir['input_ddr_addr']))]
            tmp.append([-1])
            for i in range(len(ir['input_ddr_addr'])):
                for item in ir['input_ddr_addr'][i]:
                    tmp[item].append(i+1)
                for item in ir['shortcut_source'][i]:
                    tmp[item].append(i+1)
            for i in range(len(ir['input_ddr_addr'])):
                if tmp[i] == []:
                    tmp[i] = [-1]
            ir['output_ddr_addr'] = tmp[1:]
        # compute MAdd number on OPU_IR
        self.get_mac_number(ir)
        # print 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # old version
        '''for ii in range(ir['num'][0]):
            ir['input_size'][ii] = ir['input_size'][ii][1:]
            ir['output_size'][ii] = ir['output_size'][ii][1:]
            ir['ker_size'][ii] = [ir['ker_size'][ii][1],ir['ker_size'][ii][2],ir['ker_size'][ii][0]]
            ir['ker_stride'][ii] = [ir['ker_stride'][ii][1],ir['ker_stride'][ii][2],ir['ker_stride'][ii][0]]
            ir['pooling_size'][ii] = [ir['pooling_size'][ii][1],ir['pooling_size'][ii][2],ir['pooling_size'][ii][0]]
            ir['pooling_stride'][ii] = [ir['pooling_stride'][ii][1],ir['pooling_stride'][ii][2],ir['pooling_stride'][ii][0]]
            ir['pre_padding_size'][ii] = [ir['pre_padding_size'][ii][1],ir['pre_padding_size'][ii][2],ir['pre_padding_size'][ii][0]]
        ir.pop('upsampling_scale')
        ir['post_padding'] = [0 for i in range(len(self.layer_infos))]
        ir['post_padding_size'] = [[0,0,0] for i in range(len(self.layer_infos))]  '''  
        
        with open('IR.json','w') as fp:
            json.dump(ir, fp)        
        with open(os.path.join(self.out_dir,'OPU_IR.txt'),'w') as f:
            for item in ir.keys():
                print(item,':',ir[item],sep='',file=f)
        with open(os.path.join(self.out_dir,'OPU_IR_Readable.txt'),'w') as f:
            num = ir['num'][0]
            for i in range(num):
                print('='*10, file=f)
                print('id:',i+1,file=f)
                for item in ir.keys():
                    if not item=='num': 
                        print(item,':',ir[item][i],sep='',file=f)
                        