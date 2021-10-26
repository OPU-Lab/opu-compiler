import tvm
import copy
import numpy as np

class Analysis:
    def __init__(self, nodes, params):
        self.params = params
        self.nodes = nodes
        # [x,y] x->module count, y->execution phase order, add/multiply for elementise op, bias add/mul would virtually mapped to ipa so not counted here
        self.template = { \
            'pad':[1,1], \
            'inner_product':[1,2], \
            'add':[1,3], \
            'multiply':[2,3], \
            'comparator':[1,3], \
            'pool_pad':[1,3], \
            'pooling':[1,3], \
            'interpolation':[1,4], \
        }
        self.tot_phase = 4 #np.max(np.array([item[1] for _,item in enumerate(self.template)]))+1
        self.annotate_outputs()
        #self.shuffle()
        self.traverse_op_node()
        # self.print_groups()
    
    def shuffle(self):
        transpose_nodes = [x for x in self.nodes if 'op_name' in dir(x) and x.op_name=='transpose']
        for id, node in enumerate(transpose_nodes):
            print('#'*20,id)
            tid = node.attrs['id']
            dconv_node = [x for x in self.nodes if x.attrs['id']==tid+3][0]
            conv_node = [x for x in self.nodes if x.attrs['id']==tid+11][0]
            
            groups = conv_node.op_attrs['groups']
            dw = self.params[dconv_node.inputs[1]].asnumpy()
            dw_shape = dw.shape
            print(dw_shape)
            #import ipdb
            #ipdb.set_trace()
            Kh, Kw, S = dw_shape[2],dw_shape[3],dw_shape[0]//groups
            #dw = dw.reshape([S,3,1,Kh,Kw]).transpose([1,0,2,3,4]).reshape(dw_shape)
            if S==20:
                x = dw
                tmp = np.concatenate([x[0:20:3,:,:,:],x[1:20:3,:,:,:],x[2:20:3,:,:,:],x[20:40:3,:,:,:],x[21:40:3,:,:,:],x[22:40:3,:,:,:],x[40:60:3,:,:,:],x[41:60:3,:,:,:],x[42:60:3,:,:,:]],axis=0)
                self.params[dconv_node.inputs[1]] = tvm.ndarray.array(tmp)
            elif S==40:
                x = dw
                tmp = np.concatenate([x[0:40:3,:,:,:],x[1:40:3,:,:,:],x[2:40:3,:,:,:],x[40:80:3,:,:,:],x[41:80:3,:,:,:],x[42:80:3,:,:,:],x[80:120:3,:,:,:],x[81:120:3,:,:,:],x[82:120:3,:,:,:]],axis=0)
                self.params[dconv_node.inputs[1]] = tvm.ndarray.array(tmp)
            elif S==80:
                x = dw
                tmp = np.concatenate([x[0:80:3,:,:,:],x[1:80:3,:,:,:],x[2:80:3,:,:,:],x[80:160:3,:,:,:],x[81:160:3,:,:,:],x[82:160:3,:,:,:],x[160:240:3,:,:,:],x[161:240:3],x[162:240:3,:,:,:]],axis=0)
                self.params[dconv_node.inputs[1]] = tvm.ndarray.array(tmp)
            else:
                print(dconv_node)
                import ipdb
                ipdb.set_trace()
                exit()
            
            for ix in [5,7,9]:
                bn_node = [x for x in self.nodes if x.attrs['id']==tid+ix][0]
                bn_param = self.params[bn_node.inputs[1]].asnumpy()
                x = bn_param
                print(x.shape)
                S2 = int(S*2)
                S3 = int(S*3)
                tmp = np.concatenate([x[0:S:3,:,:],x[1:S:3,:,:],x[2:S:3,:,:],x[S:S2:3,:,:],x[S+1:S2:3,:,:],x[S+2:S2:3,:,:],x[S2:S3:3,:,:],x[S2+1:S3:3,:,:],x[S2+2:S3:3,:,:]],axis=0)
                self.params[bn_node.inputs[1]] = tvm.ndarray.array(tmp)
                
            gw = self.params[conv_node.inputs[1]].asnumpy()
            gw_shape = gw.shape
            print(gw_shape)
            #import ipdb
            #ipdb.set_trace()
            Kh, Kw, S, Co = gw_shape[2],gw_shape[3],gw_shape[1],gw_shape[0]
            gw = np.concatenate([gw[:,0:S:3,:,:],gw[:,1:S:3,:,:],gw[:,2:S:3,:,:]],axis=1)
            self.params[conv_node.inputs[1]] = tvm.ndarray.array(gw)
                
            node.op_attrs.pop('axes')      
            
    
    def is_param(self, x):
        if not isinstance(x, int):
            x = x.attrs['id']
        return x in self.params
                
    def annotate_outputs(self):
        self.vc_dict = {}
        self.cv_dict = {}
        for id, node in enumerate(self.nodes):
            cid = node.attrs['id']
            self.vc_dict[id] = cid
            self.cv_dict[cid] = id
            if not self.is_param(node) and 'inputs' in dir(node):
                node.attrs['outputs'] = []
                for inp in node.inputs:
                    if self.is_param(inp):
                        continue
                    inp_node = [x for x in self.nodes if x.attrs['id']==inp][0]
                    attr = inp_node.attrs
                    #attr = self.nodes[inp].attrs
                    if 'outputs' in attr:
                        attr['outputs'].append(node.attrs['id'])
                    else:
                        attr['outputs'] = [node.attrs['id']]
    
    def _inner_product(self, op_name):
        return 'inner_product'
        
    def _comparator(self, op_name):
        return 'comparator'
        
    def _pooling(self, op_name):
        return 'pooling'
        
    def _interpolation(self, op_name):
        return 'interpolation'

    def _others(self, op_name):
        return 'others'
        
    def _ignore(self, op_name):
        return 'ignore'    
        
    def check_add_mul(self, node):
        inps = node.inputs
        inps_from_params = [1 if x in self.params else 0 for x in inps ]
        if sum(inps_from_params) == 0:
            return node.op_name
        else:
            return 'others'
    
    def check_pad(self, node):
        assert len(node.attrs['outputs'])==1, "pad node's successor number > 1"
        succ = node.attrs['outputs'][0]
        succ = self.cv_dict[succ]
        succ_op_name_mapped = self.op_map(self.nodes[succ])
        if  succ_op_name_mapped == self._inner_product(node):
            return 'pad'
        elif succ_op_name_mapped == 'pooling':
            return 'pool_pad'
        else:
            assert 0,"pad node is invalidly followed by {}".format(self.nodes[succ].op_name)    
        
                        
    def op_map(self, node):     
        if 'op_name' not in dir(node):
            return 'others'
        mapping_fn_dict = {
            # pad
            'nn.pad': self._check_op('pad'),
            # conv
            'nn.conv2d':self._inner_product,
            'nn.conv3d':self._inner_product,
            'nn.dense':self._inner_product,
            # add
            'add': self._check_op('add'),
            'nn.bias_add': self._check_op('add'),
            # multiply
            'multiply': self._check_op('multiply'),
            'divide': self._check_op('multiply'),
            # activation
            'nn.relu':self._comparator,
            'nn.leaky_relu':self._comparator,
            'clip':self._comparator,
            # pool
            'nn.max_pool2d':self._pooling,
            'nn.max_pool3d':self._pooling,
            'nn.avg_pool2d':self._pooling,
            'nn.avg_pool3d':self._pooling,
            'nn.global_avg_pool2d':self._pooling,
            'mean':self._pooling,
            # interpolation
            'nn.upsampling':self._interpolation,
            # others, virtually mapped to hardware
            'tanh':self._others,
            'strided_slice':self._others,
            'sigmoid':self._others,
            'concatenate':self._others,
            'expand_dims':self._others,
            'transpose':self._others,
            'reshape':self._others,
            'shape_of':self._ignore,
            'take':self._ignore,
            'cast':self._ignore,
            'nn.batch_flatten':self._others,
            'nn.log_softmax':self._ignore,
            'squeeze':self._others
        }
        '''if node.op_name not in mapping_fn_dict:
            return 'ignore'   ''' 
        x = mapping_fn_dict[node.op_name](node)
        # print(node.op_name,'->',x)
        return x
    
    def _check_op(self, op_name):                
        def _mapping(node):
            if op_name in ['add', 'multiply']:
                return self.check_add_mul(node)
            elif op_name == 'pad':
                return self.check_pad(node)
        return _mapping
    
    def get_phase_avail(self, phase_id):
        '''cnts = [self.pattern[item][0] for _,item in enumerate(self.pattern) if self.pattern[item][1]==phase_id]
        #cnts = [item[0] for _,item in enumerate(self.pattern) if item[1]==phase_id]
        return 0 in cnts #sum(cnts)'''
        for ii,item in enumerate(self.pattern):
            if self.pattern[item][1]==phase_id and self.pattern[item][0]<self.template[item][0]:
                return True
        return False
        
    def check_succ_avail(self, phase_id):
        sum = []
        for ii in range(phase_id+1,self.tot_phase):
            sum.append(self.get_phase_avail(ii))
            print(ii,sum[-1])
        avail = True not in sum
        return avail

    def check_layer_def(self):
        cond0 = self.pattern['inner_product'][0] == 0
        cond1 = self.pattern['pooling'][0] == 0
        cond2 = self.pattern['comparator'][0] == 0
        cond3 = self.pattern['add'][0]==0
        cond4 = self.pattern['multiply'][0] < self.template['multiply'][0]
        valid = cond4 or cond0 or cond1 or cond2
        return valid
    
    def check_pattern_valid(self, s):
        node = self.nodes[s]
        mapped_module_name = self.op_map(node)
        if mapped_module_name in self.template:
            if self.pattern[mapped_module_name][0] == 0:
                valid = False
            else:
                phase_id = self.pattern[mapped_module_name][1]
                avail_1 = self.check_succ_avail(phase_id)
                avail_2 = not mapped_module_name=='pool_pad' or (mapped_module_name == 'pool_pad' and not self.pattern['pooling'][0]==0)
                inps_in_same_group = [x for x in node.inputs if x in self.sub_group]           
                avail_3 = not (mapped_module_name=='multiply' and not len(inps_in_same_group)==len(node.inputs))
                succ_op = self.nodes[self.cv_dict[node.attrs['outputs'][0]]].op_name
                avail_4 = not (mapped_module_name=='pooling' and succ_op=='add' and node.attrs['id']+3 > node.attrs['outputs'][0])
                avail_5 = not (mapped_module_name=='pooling' and not len(inps_in_same_group)==len(node.inputs))
                #print(self.sub_group, inps_in_same_group, node.inputs)
                #print(avail_1, avail_2, avail_3, avail_4, avail_5)
                if avail_1 and avail_2 and avail_3 and avail_4 and avail_5:
                    self.pattern[mapped_module_name][0] -= 1
                    valid = True
                else:
                    valid = False
        elif mapped_module_name == 'ignore':
            valid = False
        else:
            valid = True
        if not valid:
            if self.sub_group and self.check_layer_def():
                self.groups.append(self.sub_group_c)
                if not mapped_module_name=='ignore':
                    self.sub_group = [self.vc_dict[s]]
                    self.sub_group_c = [s]
                else:
                    self.sub_group = []
                    self.sub_group_c = []
                self.pattern = copy.deepcopy(self.template)
                if mapped_module_name in self.pattern:
                    self.pattern[mapped_module_name][0] -= 1
        else: # it is possible s!=node.attrs['id'] when partial graph imported, i.e. when computing shape
            self.sub_group.append(self.vc_dict[s])
            self.sub_group_c.append(s)
        return valid
    
    def pick_next(self, queue, valid):
        if len(queue)==1:
            return queue.pop(0)
        elif not valid:
            #print('#'*10,queue)
            x = [self.cv_dict[x] for x in queue]
            x = np.argmin(np.array(x))
            return queue.pop(x)
        else:
            #print('*'*10,queue)
            x = [self.cv_dict[x] for x in queue]
            x = np.argmin(np.array(x))
            return queue.pop(x)
            #print('*'*10,queue)
            last = self.nodes[self.cv_dict[queue[-1]]]
            first = self.nodes[self.cv_dict[queue[0]]]
            if first.attrs['id'] >= last.attrs['id']:#min(last.attrs['outputs']):
                return queue.pop(-1)
            else:
                return queue.pop(0)
            
    
    
    def traverse_op_node(self):
        # assume input only has one successor
        st_id = self.nodes[0].attrs['outputs'][0]
        st_id = self.cv_dict[st_id]
        visited = []
        queue = [st_id]
        self.groups = []
        self.pattern = copy.deepcopy(self.template)
        self.sub_group = []
        self.sub_group_c = []
        valid = True
        while queue:
            s = self.pick_next(queue, valid)
            s = self.cv_dict[s]
            #if len(self.nodes)==637 and s in [617, 622]:
            #    import ipdb
            #    ipdb.set_trace()
            valid = self.check_pattern_valid(s) #self.cv_dict[s]
            #if not valid: 
            #     print('='*30)
            node = self.nodes[s]
            #print ('[TRAVERSE]',s, node)
            for i in node.attrs['outputs']: 
                if i not in visited: 
                    queue.append(i) 
                    visited.append(i)
        if self.check_layer_def():
            self.groups.append(self.sub_group_c)
    
    def print_groups(self):
        return
        for gid, group in enumerate(self.groups):
            print('='*10,gid,'='*10)
            for nid in group:
                if not nid==0:
                    print(nid, self.nodes[nid].op_name, self.nodes[nid].inputs, self.nodes[nid].attrs['outputs'])
            