import tensorflow as tf
import copy
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
import numpy as np
from collections import defaultdict

class tfGraphEditor:
    def __init__(self, graph_def):
        self.removeNodeNameList = []
        self.updateNodeNameList = []
        self.graph_def = graph_def
        self.dfs_visited = {}
        self.dfs_res = []
        #self.addOutputShape()
    
    def setGraphInput(self, inp):
        self.inp = inp
      
    def checkConv2DBackpropInput(self):# swap main input tensor(i.e. conv/relu) to 1st input
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.op=='Conv2DBackpropInput':
                tmp = [u for u in x.input]
                x.ClearField('input')
                x.input.append(tmp[2])# input
                x.input.append(tmp[1])# weight
                x.input.append(tmp[0])# shape
            out.node.extend([x])
        return out
        
      
    def checkBatchNormParam(self): # in case [] param in bn, calculate according to input data
        updateDict = {}
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(self.graph_def)
        with tf.Session(graph=graph) as sess:
            inp_tensor = sess.graph.get_tensor_by_name('import/input:0')
            feed_dict = {inp_tensor: self.inp}
            bnNode = [x for x in self.graph_def.node if 'BatchNorm' in x.op]
            for item in bnNode:
                out = [sess.graph.get_tensor_by_name('import/'+x+':0') for x in item.input]
                out = sess.run(out, feed_dict)
                if out[3].size==0:
                    mean = [np.sum(out[0].transpose(0,3,1,2)[0][x][:][:])/(out[0].shape[1]*out[0].shape[2]) for x in range(out[0].shape[3])]
                    node = self.findNodeByName(item.input[3])
                    node.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np.array(mean,dtype=np.float32)))
                    updateDict[item.input[3]] = node
                    variance = [np.sum(np.square(out[0].transpose(0,3,1,2)[0][x][:][:]-mean[x]))/(out[0].shape[1]*out[0].shape[2]) for x in range(out[0].shape[3])]
                    node = self.findNodeByName(item.input[4])
                    node.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np.array(variance,dtype=np.float32)))
                    updateDict[item.input[4]] = node
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateDict.keys():
                x=updateDict[x.name]
            out.node.extend([x])
        return out
    
    def addOutputShape(self):
        tf.import_graph_def(self.graph_def)
        sess = tf.Session()
        inp_name = [x.name for x in sess.graph.as_graph_def().node if x.op=='Placeholder'][0]
        inp = sess.graph.get_tensor_by_name(inp_name+':0')
        inp_np = np.ones([1,inp.shape[1].value,inp.shape[2].value,inp.shape[3].value])
        tlist = []
        node_names = [x.name for x in self.graph_def.node if 'size' in x.op]
        for item in node_names:
            tlist.append(sess.graph.get_tensor_by_name('import/'+item+':0'))
        tlist = sess.run(tlist,feed_dict={inp:inp_np})
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in node_names:
                idx = node_names.index(x.name)
                output_shape = tlist[idx].shape
                x.attr.get_or_create('_output_shapes')
                
                import ipdb
                ipdb.set_trace()
                print('k')
            out.node.extend([x])
        return out
    
    def checkDropOut(self):
        updateDict = {}
        dropout_nodes = [x for x in self.graph_def.node if 'dropout' in x.name]
        dropout_node_names = [x.name for x in dropout_nodes]
        dpDict = defaultdict(list)
        for item in dropout_nodes:
            group_name = [x for x in item.name.split('/') if 'dropout' in x][0]
            if group_name.startswith('^'): group_name = group_name[1:]
            dpDict[group_name].append(item)
        for key in dpDict.keys():
            # find inputs to dropout supernode
            inps = []
            outs = []
            for node in dpDict[key]:
                for inp in node.input:
                    inps.append(inp)
                outs += [x.name for x in self.graph_def.node if node.name in x.input]
            names = [x.name for x in dpDict[key]]
            dropout_inps = [x for x in set(inps) if x not in names]
            dropout_group_inps = [x for x in dropout_inps if self.findNodeByName(x) is not None and x not in dropout_node_names]
            # find outputs of dropout supernode
            dropout_outs = [x for x in set(outs) if x not in names]
            dropout_group_outs = [x for x in dropout_outs if self.findNodeByName(x) is not None and x not in dropout_node_names]
            for item in dropout_group_outs:
                node = self.findNodeByName(item)
                node_inps = [x for x in node.input if x not in dropout_node_names]
                node.ClearField('input')
                node.input.append(dropout_group_inps[0])#order matters
                for x in node_inps: node.input.append(x)
                updateDict[node.name] = node
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateDict.keys():
                out.node.extend([updateDict[x.name]])
            elif x.name in dropout_node_names:
                continue
            else:
                out.node.extend([x])
        return out
            
            
    def checkPadding_t(self):# if pool_padding==SAME, change to VALID and add paddings to conv2d in the same opu layer
        #sample graph_def that includes Pad and paddings nodes
        mpath = '/home/tiandong/tvm/example/resnet-101/resnet_v1_101_frozen.pb'
        with tf.gfile.FastGFile(mpath, 'rb') as f:
            graph_def_s = tf.GraphDef()
            graph_def_s.ParseFromString(f.read())    
            padNode_sample = [x for x in graph_def_s.node if x.name=='resnet_v1_101/Pad'][0]
            padNode_sample.ClearField('input')
            paddingNode_sample = [x for x in graph_def_s.node if x.name=='resnet_v1_101/Pad/paddings'][0]
            paddingNode_sample.ClearField('input')
            pad_ind = 0
        # check padding mode in Pooling op
        updateDict = {}
        padNodeDict = defaultdict(list)
        nodes = self.graph_def.node  
        phNodes = [x for x in nodes if x.op=='Placeholder']
        ph_shape = [x.size for x in phNodes[0].attr['shape'].shape.dim]
        poolNodes = [x for x in nodes if x.op.find('Pool')!=-1]
        poolNodeNames = [x.name+':0' for x in poolNodes]
        poolNodeInpNames = [x.input[0]+':0' for x in poolNodes]# assume pooling only has one input node
        with tf.Session(graph) as sess:
            tf.import_graph_def(self.graph_def,name='')
            inp = sess.graph.get_tensor_by_name(phNodes[0].name+':0')
            feed_dict = {inp: np.zeros(ph_shape)}
            pools = sess.run(poolNodeNames, feed_dict)
            poolInps = sess.run(poolNodeInpNames, feed_dict)
        for i in range(len(poolNodeNames)):
            x = poolNodes[i]
            pool_out_shape = pools[i].shape
            pool_in_shape = poolInps[i].shape
            stride = x.attr['strides'].list.i[1:3]
            ker_size = x.attr['ksize'].list.i[1:3]
            pad_num = [int(max(stride[0]*(pool_out_shape[1]-1)+ker_size[0]-pool_in_shape[1],0)),int(max(stride[1]*(pool_out_shape[2]-1)+ker_size[1]-pool_in_shape[2],0))]
            pad_size_0 = [int(x//2) for x in pad_num]# left, top
            pad_size = pad_size_0+[int(pad_num[0] - pad_size_0[0]),int(pad_num[1] - pad_size_0[1])]
            if sum(pad_size)!=0:
                # get conv2d in the same opu layer by looking forward
                self.dfs_visited = {}
                self.dfs_res.clear()
                self.dfs(x, True)
                # irregular padding in single pool can be dealt, new opu layer
                if not self.dfs_res[0].op=='Conv2D': continue
                x.attr['padding'].s=b'VALID'
                updateDict[x.name] = x
                convNode = self.dfs_res[0]
                padNode = [self.findNodeByName(i) for i in convNode.input if i.find('Pad')!=-1]
                if len(padNode)>0:
                    padNode = padNode[0]
                    paddingNode = [self.findNodeByName(i) for i in padNode.input if i.find('paddings')!=-1]
                    paddingNode = paddingNode[0]
                    for key,value in paddingNode.attr.items():
                        if key=="value":
                            np_array = tensor_util.MakeNdarray(value.tensor)
                            np_array[1] = [np_array[1][0]+pad_size[0], np_array[1][1]+pad_size[2]]
                            np_array[2] = [np_array[2][0]+pad_size[1], np_array[2][1]+pad_size[3]]
                            paddingNode.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                    updateDict[paddingNode.name] = paddingNode
                else:
                    # No existing pad in the graph_def being analyzed, insert pad node with reference to sample pad node
                    paddingNode_t = copy.deepcopy(paddingNode_sample)
                    paddingNode_t.name = 'paddings_'+str(pad_ind)
                    for key,value in paddingNode_t.attr.items():
                        if key=="value":
                            np_array = tensor_util.MakeNdarray(value.tensor)
                            np_array[1] = [pad_size[0], pad_size[2]]
                            np_array[2] = [pad_size[1], pad_size[3]]
                            paddingNode_t.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                    padNode_t = copy.deepcopy(padNode_sample)
                    padNode_t.name = 'pad_'+str(pad_ind)
                    pad_ind+=1
                    padNode_t.input.append(convNode.input[0])
                    padNode_t.input.append(paddingNode_t.name)
                    padNodeDict[convNode.name].append(paddingNode_t)
                    padNodeDict[convNode.name].append(padNode_t)
                    inps = convNode.input
                    convNode.ClearField('input')
                    convNode.input.append(padNode_t.name)
                    convNode.input.append(inps[1])
                    if '_output_shapes' in convNode.attr.keys():
                        convNode.attr.pop('_output_shapes')
                    updateDict[convNode.name] = convNode
                    #import ipdb
                    #ipdb.set_trace()
                    #print()
                #import ipdb
                #ipdb.set_trace()
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateDict.keys():
                for item in padNodeDict[x.name]:
                    out.node.extend([item])
                out.node.extend([updateDict[x.name]])
            else:
                # since we have changed tensor shape of pool/conv, '_output_shapes' leads to shape inconsistency error, so remove this attribute in graph_def
                if '_output_shapes' in x.attr.keys():
                    x.attr.pop('_output_shapes')
                out.node.extend([x])
        return out            
            
                
            
    
    
    def checkPadding(self):# if pool_padding==SAME, change to VALID and add paddings to conv2d in the same opu layer
        #sample graph_def that includes Pad and paddings nodes
        mpath = '/home/tiandong/tvm/example/resnet-101/resnet_v1_101_frozen.pb'
        with tf.gfile.FastGFile(mpath, 'rb') as f:
            graph_def_s = tf.GraphDef()
            graph_def_s.ParseFromString(f.read())    
            padNode_sample = [x for x in graph_def_s.node if x.name=='resnet_v1_101/Pad'][0]
            padNode_sample.ClearField('input')
            paddingNode_sample = [x for x in graph_def_s.node if x.name=='resnet_v1_101/Pad/paddings'][0]
            paddingNode_sample.ClearField('input')
            pad_ind = 0
        # check padding mode in Pooling op
        updateDict = {}
        padNodeDict = defaultdict(list)
        nodes = self.graph_def.node  
        phNodes = [x for x in nodes if x.op=='Placeholder']
        ph_shape = [x.size for x in phNodes[0].attr['shape'].shape.dim]
        poolNodes = [x for x in nodes if x.op.find('Pool')!=-1]
        poolNodeNames = [x.name+':0' for x in poolNodes]
        poolNodeInpNames = [x.input[0]+':0' for x in poolNodes]# assume pooling only has one input node
        with tf.Session(graph) as sess:
            tf.import_graph_def(self.graph_def,name='')
            inp = sess.graph.get_tensor_by_name(phNodes[0].name+':0')
            feed_dict = {inp: np.zeros(ph_shape)}
            pools = sess.run(poolNodeNames, feed_dict)
            poolInps = sess.run(poolNodeInpNames, feed_dict)
        for i in range(len(poolNodeNames)):
            x = poolNodes[i]
            pool_out_shape = pools[i].shape
            pool_in_shape = poolInps[i].shape
            stride = x.attr['strides'].list.i[1:3]
            ker_size = x.attr['ksize'].list.i[1:3]
            pad_num = [int(max(stride[0]*(pool_out_shape[1]-1)+ker_size[0]-pool_in_shape[1],0)),int(max(stride[1]*(pool_out_shape[2]-1)+ker_size[1]-pool_in_shape[2],0))]
            pad_size_0 = [int(x//2) for x in pad_num]# left, top
            pad_size = pad_size_0+[int(pad_num[0] - pad_size_0[0]),int(pad_num[1] - pad_size_0[1])]
            if sum(pad_size)!=0:
                # get conv2d in the same opu layer by looking forward
                self.dfs_visited = {}
                self.dfs_res.clear()
                self.dfs(x, True)
                # irregular padding in single pool can be dealt, new opu layer
                if not self.dfs_res[0].op=='Conv2D': continue
                x.attr['padding'].s=b'VALID'
                updateDict[x.name] = x
                convNode = self.dfs_res[0]
                padNode = [self.findNodeByName(i) for i in convNode.input if i.find('Pad')!=-1]
                if len(padNode)>0:
                    padNode = padNode[0]
                    paddingNode = [self.findNodeByName(i) for i in padNode.input if i.find('paddings')!=-1]
                    paddingNode = paddingNode[0]
                    for key,value in paddingNode.attr.items():
                        if key=="value":
                            np_array = tensor_util.MakeNdarray(value.tensor)
                            np_array[1] = [np_array[1][0]+pad_size[0], np_array[1][1]+pad_size[2]]
                            np_array[2] = [np_array[2][0]+pad_size[1], np_array[2][1]+pad_size[3]]
                            paddingNode.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                    updateDict[paddingNode.name] = paddingNode
                else:
                    # No existing pad in the graph_def being analyzed, insert pad node with reference to sample pad node
                    paddingNode_t = copy.deepcopy(paddingNode_sample)
                    paddingNode_t.name = 'paddings_'+str(pad_ind)
                    for key,value in paddingNode_t.attr.items():
                        if key=="value":
                            np_array = tensor_util.MakeNdarray(value.tensor)
                            np_array[1] = [pad_size[0], pad_size[2]]
                            np_array[2] = [pad_size[1], pad_size[3]]
                            paddingNode_t.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                    padNode_t = copy.deepcopy(padNode_sample)
                    padNode_t.name = 'pad_'+str(pad_ind)
                    pad_ind+=1
                    padNode_t.input.append(convNode.input[0])
                    padNode_t.input.append(paddingNode_t.name)
                    padNodeDict[convNode.name].append(paddingNode_t)
                    padNodeDict[convNode.name].append(padNode_t)
                    inps = convNode.input
                    convNode.ClearField('input')
                    convNode.input.append(padNode_t.name)
                    convNode.input.append(inps[1])
                    if '_output_shapes' in convNode.attr.keys():
                        convNode.attr.pop('_output_shapes')
                    updateDict[convNode.name] = convNode
                    #import ipdb
                    #ipdb.set_trace()
                    #print()
                #import ipdb
                #ipdb.set_trace()
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateDict.keys():
                for item in padNodeDict[x.name]:
                    out.node.extend([item])
                out.node.extend([updateDict[x.name]])
            else:
                # since we have changed tensor shape of pool/conv, '_output_shapes' leads to shape inconsistency error, so remove this attribute in graph_def
                if '_output_shapes' in x.attr.keys():
                    x.attr.pop('_output_shapes')
                out.node.extend([x])
        return out    
    
    def dfs(self, src_node, contd):
        if not contd: return
        for iNodeName in src_node.input:
            if iNodeName not in self.dfs_visited.keys():
                self.dfs_visited[iNodeName] = True
                inode = self.findNodeByName(iNodeName)
                if inode.op=='Conv2D' or inode.op=='Concat' or inode.op=='ConcatV2' or inode.op=='AvgPool' or inode.op=='MaxPool':
                    contd = False
                    self.dfs_res.append(inode)
                else:
                    contd = True
                self.dfs(inode, contd)
         
        
    def setNodeAsPlaceholder(self, node_name, shape=None):
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name==node_name:
                sample = [x for x in self.graph_def.node if x.op=='Placeholder']
                temp = copy.deepcopy(sample[0])
                temp.name = x.name
                temp.attr['dtype'].type = x.attr['T'].type
                if shape is not None:
                    for i in range(len(temp.attr['shape'].shape.dim)):
                        temp.attr['shape'].shape.dim[i].size = shape[i]
                else:
                    if '_output_shapes' in temp.attr.keys():
                        temp.attr['_output_shapes'].CopyFrom(x.attr['_output_shapes'])
                    if 'shape' in temp.attr.keys():
                        temp.attr['shape'].shape.CopyFrom(x.attr['_output_shapes'].list.shape[0])
                out.node.extend([temp])
            else:
                out.node.extend([x])
        return out   
        
    def setInputBatchSize(self, bz=1):
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.op=="Placeholder":
                x.attr['shape'].shape.dim[0].size=bz
                out.node.extend([x])
            else:
                out.node.extend([x])
        return out   
        
    def changeLeakyReluAlpha(self, alpha_new=0.125):
        nodes = self.graph_def.node     
        leakyReluNodes = [x for x in nodes if x.op=="Maximum"]
        leakyAlphaNames = []
        for item in leakyReluNodes:
            for x in item.input:
                if "mul" in x:
                    leakyAlphaNames.append(x)
        leakyAlphaNodes = [self.findNodeByName(x) for x in leakyAlphaNames]
        alphaNames = []
        for item in leakyAlphaNodes:
            for x in item.input:
                if "mul" in x:
                    alphaNames.append(x)
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in alphaNames:
                for key,value in x.attr.items():
                    if key=="value":
                        np_array = tensor_util.MakeNdarray(value.tensor)
                np_array = np.array(alpha_new,dtype=np_array.dtype)
                x.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                out.node.extend([x])
            else:
                out.node.extend([x])
            
        return out   
        
    def change1stConvDimOrder(self, firstConvName):
        nodes = self.graph_def.node     
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name == firstConvName:
                for key,value in x.attr.items():
                    if key=="value":
                        np_array = tensor_util.MakeNdarray(value.tensor)
                np_array_new = np_array[:,:,::-1,:]
                x.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array_new))
                out.node.extend([x])
            else:
                out.node.extend([x])
        return out 
     
    def changePoolSize(self):
        nodes = self.graph_def.node     
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name == '17-maxpool':
                x.attr['padding'].s=b'VALID'
                out.node.extend([x])
            else:
                out.node.extend([x])
        pnode = [x for x in self.graph_def.node if x.name=='17-maxpool'][0]
        return out 
        
    def change6thConv(self):
        nodes = self.graph_def.node     
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name == 'Pad_5/paddings':
                for key,value in x.attr.items():
                    if key=="value":
                        np_array = tensor_util.MakeNdarray(value.tensor)
                np_array = np.array([[0,0],[1,2],[1,2],[0,0]],dtype=np.int32)
                x.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
                out.node.extend([x])
            else:
                out.node.extend([x])
        pnode = [x for x in self.graph_def.node if x.name=='Pad_5/paddings'][0]
        return out 
       
    def fuseBN(self):
        removeNodeNameList = []
        updateNodeDict = {}
        bnParams = {}
        # assume bn usually appears right after conv2d
        conv2dNodeNameList = [x.name for x in self.graph_def.node if x.op=="Conv2D"]
        afterConvNodes = []
        for x in self.graph_def.node:
            for inp in x.input:
                if inp in conv2dNodeNameList:
                    afterConvNodes.append(x)
        for x in afterConvNodes:
            if x.op == "Sub": # bn(moving_mean, variance, beta, gamma)
                inp_conv_name = [u for u in x.input if u in conv2dNodeNameList][0]
                mean_name = [u for u in x.input if u!=inp_conv_name][0]
                afterSubNode = [u for u in self.graph_def.node if x.name in u.input][0]
                variance_name = [u for u in afterSubNode.input if u!=x.name][0]
                afterDivNode = [u for u in self.graph_def.node if afterSubNode.name in u.input][0]
                gamma_name = [u for u in afterDivNode.input if u!=afterSubNode.name][0]
                afterMulNode = [u for u in self.graph_def.node if afterDivNode.name in u.input][0]
                beta_name = [u for u in afterMulNode.input if u!=afterDivNode.name][0]
                # moving_variance = denominator^2
                varNode = self.findNodeByName(variance_name)
                for key,value in varNode.attr.items():
                    if key=="value":
                        np_array = tensor_util.MakeNdarray(value.tensor)
                np_array_sq = np.square(np_array)-1e-5
                varNode.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array_sq))
                #updateNodeDict[variance_name] = varNode
                # make BN node
                x.name = afterMulNode.name
                x.op = "BatchNormWithGlobalNormalization"
                x.ClearField('input') # (data, gamma, beta, moving_mean, moving_variance)
                x.input.append(inp_conv_name)
                x.input.append(mean_name)
                x.input.append(variance_name)
                x.input.append(beta_name)
                x.input.append(gamma_name)
                updateNodeDict[afterMulNode.name] = x
                removeNodeNameList.append(afterSubNode.name)
                removeNodeNameList.append(afterDivNode.name)
                removeNodeNameList.append(x.name)
                bnParams[x.name] = [self.findNodeByName(mean_name), varNode, self.findNodeByName(beta_name), self.findNodeByName(gamma_name)]
                removeNodeNameList.append(mean_name)
                removeNodeNameList.append(variance_name)
                removeNodeNameList.append(beta_name)
                removeNodeNameList.append(gamma_name)
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateNodeDict.keys():
                if x.name in bnParams.keys():
                    for t in bnParams[x.name]:
                        out.node.extend([t])
                out.node.extend([updateNodeDict[x.name]])
            elif x.name not in removeNodeNameList:
                out.node.extend([x])
        self.graph_def = out
        return out
                
    def getVal(self,varNode):
        for key,value in varNode.attr.items():
            if key=="value":
                np_array = tensor_util.MakeNdarray(value.tensor) 
        return np_array  
        
    def alterConv2dDataLayout(self): #data: NCHW(tvm transpose kernel) <- NHWC(tvm add transpose to input) #no need actually -- user can specify layout for input
        conv2dNodes = [x for x in self.graph_def.node if x.op=="Conv2D"]
        filterNodeNameList = []
        for x in conv2dNodes:
            filter_name = [t for t in x.input if t.find("filter")!=-1 or t.find("kernel")!=-1][0]
            filterNodeNameList.append(filter_name)
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.op=="Conv2D":
                x.attr['data_format'].s = b'NHWC'
                out.node.extend([x])
            elif x.name in filterNodeNameList:
                for key,value in x.attr.items():
                    if key=="value":
                        np_array = tensor_util.MakeNdarray(value.tensor)
                np_array_tr = np.transpose(np_array,(0,1,2,3))
                print(np_array_tr.shape)
                x.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array_tr))
                out.node.extend([x])
            else:
                out.node.extend([x])
        self.graph_def = out
        return out

    def fuseActivation(self, name="LeakyRelu", new_alpha=None):
        if name=="LeakyRelu" and new_alpha is not None:
            self.graph_def = self.changeLeakyReluAlpha(alpha_new=new_alpha)
        updateNodeList = [x for x in self.graph_def.node if x.op=="Maximum"] 
        updateNodeDict = {}
        removeNodeNameList = []
        for x in updateNodeList:
            if name=="LeakyRelu":
                node_new, mul_br = self.update2LeakyReluNode(x)
            elif name=="Relu":
                node_new, mul_br = self.update2ReluNode(x)
            else:
                assert 0,"Activation Fusion Unsupported yet"
            removeNodeNameList.append(mul_br.name)
            updateNodeDict[x.name] = node_new
        out = graph_pb2.GraphDef()
        for x in self.graph_def.node:
            if x.name in updateNodeDict.keys():
                out.node.extend([updateNodeDict[x.name]])
            elif x.name not in removeNodeNameList:
                out.node.extend([x])
        self.graph_def = out
        return out 
     
    def update2ReluNode(self, x):
        inp0 = self.findNodeByName(x.input[0])
        inp1 = self.findNodeByName(x.input[1])
        if inp0.op=="Const":
            relu_factor = inp0
            direct_br = inp1
        else:
            relu_factor = inp1
            direct_br = inp0
        # make new node
        node_new = x
        node_new.op = "Relu"
        node_new.ClearField('input')
        node_new.input.append(direct_br.name)
        return node_new, relu_factor
            
    def update2LeakyReluNode(self, x):
        inp0 = self.findNodeByName(x.input[0])
        inp1 = self.findNodeByName(x.input[1])
        if x.input[0].find("mul")!=-1:
            mul_br = inp0
            direct_br = inp1
        else:
            mul_br = inp1
            direct_br = inp0
        alpha_name = [x for x in mul_br.input if not x==direct_br.name][0]
        # make new node
        node_new = x
        node_new.op = "LeakyRelu"
        node_new.ClearField('input')
        node_new.input.append(direct_br.name)
        node_new.input.append(alpha_name)
        return node_new, mul_br
        
    def findNodeByName(self, name):
        xd = [x for x in self.graph_def.node if x.name==name]
        if len(xd)==0:return None
        else: return xd[0]
        
        
    def node_to_placeholder(self, node):
        sample = [x for x in self.graph_def.node if x.op=='Placeholder']
        out = copy.deepcopy(sample[0])
        out.name = node.name
        out.attr['dtype'].type = node.attr['T'].type
        t = self.graph.get_tensor_by_name(node.name+':0')
        for i in range(4):
            ts = t.shape[i].value
            if isinstance(ts, int):
                out.attr['shape'].shape.dim[i].size = ts
            else:
                out.attr['shape'].shape.dim[i].size = -1
        return out
    