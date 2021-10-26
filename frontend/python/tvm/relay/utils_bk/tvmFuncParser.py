from __future__ import absolute_import
import json
from collections import defaultdict
import attr
from ..op import Op
from ..expr import Function, GlobalVar, Call
from ..expr_functor import ExprFunctor
from ..ty import TupleType, TensorType
from ... import target as _target
import numpy as np
@attr.s
class NodeRef(object):
    """A reference to a node, used for constructing the graph."""
    ident = attr.ib()
    index = attr.ib(default=0)
    version = attr.ib(default=0)

    def to_json(self):
        return [self.ident, self.index, self.version]


@attr.s
class Node(object):
    """The base class for nodes in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()

    def to_json(self):
        raise Exception("Abstract method, please implement me.")


@attr.s
class InputNode(Node):
    """An input node in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()

    def to_json(self):
        return {
            "op": "null",
            "name": self.name,
            "inputs": []
        }


@attr.s
class OpNode(Node):
    """An operator node in the TVM runtime system"s graph input."""
    op_name = attr.ib()
    inputs = attr.ib()
    op_attrs = attr.ib()
    num_outputs = attr.ib(default=1)

    def to_json(self):
        attrs = dict.copy(self.op_attrs)
        # Extend ops with extra info.
        attrs["func_name"] = self.op_name
        attrs["flatten_data"] = "0"
        attrs["num_inputs"] = str(len(self.inputs))
        attrs["num_outputs"] = str(self.num_outputs)

        return {
            "op": "tvm_op",
            "name": self.name,
            "attrs": attrs,
            "inputs": self.inputs
        }
def shape_to_json(shape):
    """Convert symbolic shape to json compatible forma."""
    return [sh.value for sh in shape]
class Parser(ExprFunctor):
    nodes = attr.ib()
    var_map = attr.ib()
    def __init__(self, tvmFunc):
        ExprFunctor.__init__(self)
        self.func = tvmFunc
        self.nodes = []
        self.var_map = {}
        self.params = {}
        self._name_map = {}
        self.nodeOpDict = {} # node_id -> op_name
        self.fusePairs = {} # node_id -> node_id(successor)
        self.idNodeMap = {} # node_id -> node
        self.idOutputMap = defaultdict(list) # node_id -> node_id(output)
    
    def collectNodes(self):
        for param in self.func.params:
            node = InputNode(param.name_hint, {})
            self.var_map[param] = self.add_node(node, param)
        self.visit(self.func.body)
        return self.nodes, self.params

    def check_1(self):
        #return 
        for item in self.nodes:
            print(item)
        remove_list = []
        '''extra pass to merge params for add-multiply-add, which is not folded by relay pass fold_constant'''
        for inp_id in sorted(self.fusePairs.keys()):
            cur_id = self.fusePairs[inp_id]
            #remove_list.append(cur_id)
            print('[CHECK]',inp_id, cur_id)
            inp_opname = self.nodeOpDict[inp_id]
            cur_opname = self.nodeOpDict[cur_id]
            if cur_opname=='multiply':
                remove_list.append(cur_id)
                inp_id = self.idNodeMap[cur_id].inputs[0]
                inp_opname = self.nodeOpDict[inp_id]
                if inp_opname=='add':
                    # get 3 params
                    inp_param_id = self.idNodeMap[inp_id].inputs[1]
                    inp_param = self.params[inp_param_id]
                    cur_param_id = self.idNodeMap[cur_id].inputs[1]
                    cur_param = self.params[cur_param_id]
                    conv_node_id = self.idNodeMap[inp_id].inputs[0]
                    conv_param_id = self.idNodeMap[conv_node_id].inputs[1]
                    conv_param = self.params[conv_param_id]
                    # fuse to inp_param & conv_param
                    mul_factor = np.expand_dims(cur_param.asnumpy(),3)
                    fused_conv_param = conv_param.asnumpy()*mul_factor
                    self.params[conv_param_id].copyfrom(fused_conv_param)
                    fused_add_param = inp_param.asnumpy()*cur_param.asnumpy()
                    self.params[inp_param_id].copyfrom(fused_add_param)
                    print("mul->conv_bias#",cur_param_id,'->',conv_param_id,inp_param_id)
                    # update successor's input
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_id
                elif inp_opname=='nn.leaky_relu' or inp_opname=='nn.relu':
                    # get precessor's conv bias and weight
                    conv_bias_node_id = self.idNodeMap[inp_id].inputs[0]
                    conv_bias_id = self.idNodeMap[conv_bias_node_id].inputs[1]
                    conv_bias = self.params[conv_bias_id]
                    conv_node_id = self.idNodeMap[conv_bias_node_id].inputs[0]
                    conv_w_id = self.idNodeMap[conv_node_id].inputs[1]
                    conv_w = self.params[conv_w_id]
                    # get mul param
                    cur_param_id = self.idNodeMap[cur_id].inputs[1]
                    cur_param = self.params[cur_param_id]
                    if cur_param.asnumpy()[0][0][0]<0:
                        print('mul param < 0')
                        exit()
                    # abs
                    cur_param_abs = abs(cur_param.asnumpy())
                    # fuse abs(mul) to precessor conv weight and bias
                    mul_factor = np.expand_dims(cur_param_abs,3)
                    fused_conv_w = conv_w.asnumpy()*mul_factor
                    self.params[conv_w_id].copyfrom(fused_conv_w)
                    fused_conv_bias = conv_bias.asnumpy()*cur_param_abs
                    self.params[conv_bias_id].copyfrom(fused_conv_bias)
                    print("mul->leaky_relu->conv_w#",cur_param_id,'->',conv_w_id,conv_bias_id)
                    # update successor's input
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_id
                else: # directly follows Conv2D
                    print('[WARNING] How to fuse? \'multiply\' follows',inp_opname)
                    exit()
        for inp_id in sorted(self.fusePairs.keys()):
            cur_id = self.fusePairs[inp_id]
            cur_opname = self.nodeOpDict[cur_id]
            inp_id = self.idNodeMap[cur_id].inputs[0]
            inp_opname = self.nodeOpDict[inp_id]
            if cur_opname=='add':
                print(cur_id,cur_opname,'->',inp_id,inp_opname)
        #import ipdb
        #ipdb.set_trace()
        self.nodes = []
        for idx in sorted(self.idNodeMap.keys()):
            self.nodes.append(self.idNodeMap[idx])
        remove_list.reverse()
        for idx in remove_list:
            self.nodes.pop(idx)    
        
    def check(self):
        return
        #import ipdb
        #ipdb.set_trace()
        for item in self.nodes:
            print(item)
        remove_list = []
        '''extra pass to merge params for add-multiply-add, which is not folded by relay pass fold_constant'''
        for inp_id in sorted(self.fusePairs.keys()):
            cur_id = self.fusePairs[inp_id]
            remove_list.append(cur_id)
            print('[CHECK]',inp_id, cur_id)
            inp_opname = self.nodeOpDict[inp_id]
            cur_opname = self.nodeOpDict[cur_id]
            if cur_opname=='add':
                cur_param_id = self.idNodeMap[cur_id].inputs[1]
                cur_param = self.params[cur_param_id]
                inp_id = self.idNodeMap[cur_id].inputs[0]
                inp_opname = self.nodeOpDict[inp_id]
                # update successor's input
                if inp_opname=='add':
                    inp_param_id = self.idNodeMap[inp_id].inputs[1]
                    inp_param = self.params[inp_param_id]
                    # fuse to inp_param
                    fused_param = inp_param.asnumpy()+cur_param.asnumpy()
                    self.params[inp_param_id].copyfrom(fused_param)
                    print('add->bias#',cur_param_id,'->',inp_param_id)
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_id
                elif inp_opname=='multiply':
                    inp_inp_id = self.idNodeMap[inp_id].inputs[0]  
                    inp_param_id = self.idNodeMap[inp_inp_id].inputs[1]
                    inp_param = self.params[inp_param_id]
                    # fuse to inp_param
                    fused_param = inp_param.asnumpy()+cur_param.asnumpy()
                    self.params[inp_param_id].copyfrom(fused_param)
                    print('add->mul->bias#',cur_param_id,'->',inp_param_id)                       
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_inp_id
                elif inp_opname=='nn.leaky_relu':
                    print('[WARNING] How to fuse? \'add\' follows',inp_opname)
                    exit()
            elif cur_opname=='multiply':
                inp_id = self.idNodeMap[cur_id].inputs[0]
                inp_opname = self.nodeOpDict[inp_id]
                if inp_opname=='add':
                    # get 3 params
                    inp_param_id = self.idNodeMap[inp_id].inputs[1]
                    inp_param = self.params[inp_param_id]
                    cur_param_id = self.idNodeMap[cur_id].inputs[1]
                    cur_param = self.params[cur_param_id]
                    conv_node_id = self.idNodeMap[inp_id].inputs[0]
                    conv_param_id = self.idNodeMap[conv_node_id].inputs[1]
                    conv_param = self.params[conv_param_id]
                    # fuse to inp_param & conv_param
                    mul_factor = np.expand_dims(cur_param.asnumpy(),3)
                    fused_conv_param = conv_param.asnumpy()*mul_factor
                    self.params[conv_param_id].copyfrom(fused_conv_param)
                    fused_add_param = inp_param.asnumpy()*cur_param.asnumpy()
                    self.params[inp_param_id].copyfrom(fused_add_param)
                    print("mul->conv_bias#",cur_param_id,'->',conv_param_id,inp_param_id)
                    # update successor's input
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_id
                elif inp_opname=='nn.leaky_relu':
                    # get precessor's conv bias and weight
                    conv_bias_node_id = self.idNodeMap[inp_id].inputs[0]
                    conv_bias_id = self.idNodeMap[conv_bias_node_id].inputs[1]
                    conv_bias = self.params[conv_bias_id]
                    conv_node_id = self.idNodeMap[conv_bias_node_id].inputs[0]
                    conv_w_id = self.idNodeMap[conv_node_id].inputs[1]
                    conv_w = self.params[conv_w_id]
                    # get mul param
                    cur_param_id = self.idNodeMap[cur_id].inputs[1]
                    cur_param = self.params[cur_param_id]
                    if cur_param.asnumpy()[0][0][0]<0:
                        print('mul param < 0')
                        exit()
                    # abs
                    cur_param_abs = abs(cur_param.asnumpy())
                    # fuse abs(mul) to precessor conv weight and bias
                    mul_factor = np.expand_dims(cur_param_abs,3)
                    fused_conv_w = conv_w.asnumpy()*mul_factor
                    self.params[conv_w_id].copyfrom(fused_conv_w)
                    fused_conv_bias = conv_bias.asnumpy()*cur_param_abs
                    self.params[conv_bias_id].copyfrom(fused_conv_bias)
                    print("mul->leaky_relu->conv_w#",cur_param_id,'->',conv_w_id,conv_bias_id)
                    # update successor's input
                    for i in self.idOutputMap[cur_id]:
                        self.idNodeMap[i].inputs[0] = inp_id
                else: # directly follows Conv2D
                    print('[WARNING] How to fuse? \'multiply\' follows',inp_opname)
                    exit()
            else:
                print('[ERROR] Op to be fused is neither \'add\' nor \'multiply\'')
                exit()
        import ipdb
        ipdb.set_trace()
        self.nodes = []
        for idx in sorted(self.idNodeMap.keys()):
            self.nodes.append(self.idNodeMap[idx])
        remove_list.reverse()
        for idx in remove_list:
            self.nodes.pop(idx)
 
        '''get output(s)'''
        self.output_node_ids = [x.attrs['id'] for x in self.nodes if x.attrs['id'] not in self.idOutputMap.keys()]
        
    def visit_call(self, call):
        """Transform a ::tvm.relay.Call into an operator in the TVM graph."""
        if isinstance(call.op, Op):
            raise Exception(
                "Operators should be transformed away; try applying" +
                "the fuse_ops transformation to the expression.")
        elif isinstance(call.op, GlobalVar):
            func = self.mod[call.op]
        elif isinstance(call.op, Function):
            func = call.op
        else:
            raise Exception(
                "TVM runtime does not support calls to {0}".format(type(call.op)))
        if int(func.attrs.Primitive) != 1:
            raise Exception(
                "TVM only support calls to primitive functions " +
                "(i.e functions composed of fusable operator invocations)")
        inputs = []
        # flatten tuple in the call.
        for arg in call.args:
            res = self.visit(arg)
            if isinstance(arg.checked_type, TupleType):
                assert isinstance(res, tuple)
                inputs += res
            else:
                inputs.append(res)
        op_attrs = self.getAttr(func.body)
        inputs = [x.ident for x in inputs]
        op_name = self.getFuncName(func)
        op_node = OpNode(self._get_unique_name(op_name), {},
                         op_name, inputs, op_attrs)
        return self.add_node(op_node, call)
    
    def getAttr(self, call):
        attrs = {}
        if 'op' not in dir(call): 
            return attrs
        op_name = call.op.name
        if op_name == "nn.max_pool2d" or op_name == "nn.avg_pool2d" or op_name == "nn.max_pool3d" or op_name=="nn.avg_pool3d": #nn.max_pool2d(%p0, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0], layout="NHWC"
            attrs["pool_size"] = call.attrs.__getattr__('pool_size')
            attrs["strides"] = call.attrs.__getattr__('strides')
            attrs["padding"] = call.attrs.__getattr__('padding')
            attrs["layout"] = call.attrs.__getattr__('layout')
        elif op_name == "nn.pad":#nn.pad(%input, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]]) 
            attrs["pad_width"] = call.attrs.__getattr__('pad_width')
        elif op_name == "nn.conv2d":#nn.conv2d(%0, %1, channels=16, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO")
            attrs["channels"] = call.attrs.__getattr__('channels')
            attrs["kernel_size"] = call.attrs.__getattr__('kernel_size')
            attrs["data_layout"] = call.attrs.__getattr__('data_layout')
            attrs["kernel_layout"] = call.attrs.__getattr__('kernel_layout')
            if 'groups' in dir(call.attrs):
                attrs['groups'] = call.attrs.__getattr__('groups')
        elif op_name == "nn.conv3d":
            attrs["strides"] = call.attrs.__getattr__('strides')
            attrs["padding"] = call.attrs.__getattr__('padding')
            attrs["dilation"] = call.attrs.__getattr__('dilation')
            attrs["kernel_size"] = call.attrs.__getattr__('kernel_size')
        elif op_name == "nn.leaky_relu":#nn.leaky_relu(%4, alpha=0.1)
            attrs["alpha"] = call.attrs.__getattr__('alpha')
        elif op_name == "clip":
            attrs['min'] = call.attrs.__getattr__('a_min')
            attrs['max'] = call.attrs.__getattr__('a_max')
        elif op_name == "nn.upsampling":
            attrs['scale'] = call.attrs.__getattr__('scale_h')
        elif op_name == "image.resize":
            attrs['size'] = call.attrs.__getattr__('size')
            if 'dilation_size' in dir(call.attrs):
                attrs['dilation_size'] = call.attrs.__getattr__('dilation_size')
            attrs['layout'] = call.attrs.__getattr__('layout')
            attrs['method'] = call.attrs.__getattr__('method')
        elif op_name == "strided_slice":
            attrs['begin'] = call.attrs.__getattr__('begin')
            attrs['end'] = call.attrs.__getattr__('end')
        elif op_name == "reshape":
            attrs['newshape'] = call.attrs.__getattr__('newshape')
        elif op_name == "yolo_reorg":
            attrs['newshape'] = call.attrs.__getattr__('newshape')
        elif op_name == "concatenate":
            attrs['axis'] = call.attrs.__getattr__('axis')
        elif op_name == "transpose":
            attrs['axes'] = call.attrs.__getattr__('axes')
        elif op_name == "squeeze":
            attrs['axis'] = call.attrs.__getattr__('axis')
        elif op_name == "expand_dims":
            attrs['axis'] = call.attrs.__getattr__('axis')
            attrs['num_newaxis'] = call.attrs.__getattr__('num_newaxis') 
        return attrs
            
        
    def getFuncName(self, func):
        name = []
        expr = func.body
        while isinstance(expr, Call):
            name.append(expr.op.name)
            expr = expr.args[0]
        name.reverse()
        if len(name)!=1:
            funcName = "Fused-"+"-".join(name)
        else:
            funcName = name[0]
        return funcName
        
    def visit_tuple(self, vtuple):
        fields = []
        for field in vtuple.fields:
            ref = self.visit(field)
            assert isinstance(ref, NodeRef)
            fields.append(ref)
        return tuple(fields)

    def visit_tuple_getitem(self, op):
        vtuple = self.visit(op.tuple_value)
        assert isinstance(vtuple, tuple)
        return vtuple[op.index]

    def visit_constant(self, op):
        index = len(self.params)
        name = "p%d" % index
        self.params[len(self.nodes)] = op.data
        node = InputNode(name, {})
        return self.add_node(node, op)

    def visit_function(self, _):
        raise RuntimeError("function not supported")

    def visit_if(self, _):
        raise RuntimeError("if not supported")

    def visit_global_var(self, _):
        raise RuntimeError()

    def visit_let(self, let):
        """
        Visit the let binding, by first traversing its value,
        then setting the metadata on the returned NodeRef.

        Finally visit the body, and return the NodeRef corresponding
        to it.

        Parameters
        ----------
        let: tvm.relay.Expr
            The let binding to transform.

        Returns
        -------
        ref: NodeRef
            The node reference to the body.
        """
        assert let.var not in self.var_map
        self.var_map[let.var] = self.visit(let.value)
        return self.visit(let.body)

    def visit_var(self, rvar):
        return self.var_map[rvar]
        
    def visit_op(self, _):
        raise Exception("can not compile op in non-eta expanded form")

    def visit_ref_create(self, _):
        raise RuntimeError("reference not supported")

    def visit_ref_read(self, _):
        raise RuntimeError("reference not supported")

    def visit_ref_write(self, _):
        raise RuntimeError("reference not supported")

    def visit_constructor(self, _):
        raise Exception("ADT constructor case not yet implemented")

    def visit_match(self, _):
        raise Exception("match case not yet implemented")
        
    def add_node(self, node, expr):
        checked_type = expr.checked_type
        node_id = len(self.nodes)
        self.nodes.append(node)
        # Tuple return value, flatten as tuple
        if isinstance(checked_type, TupleType):
            ret = []
            shape = []
            dtype = []
            for i, typ in enumerate(checked_type.fields):
                if not isinstance(typ, TensorType):
                    raise RuntimeError("type %s not supported" % typ)
                ret.append(NodeRef(node_id, i))
                shape.append(shape_to_json(typ.shape))
                dtype.append(typ.dtype)
            node.attrs["shape"] = shape
            node.attrs["dtype"] = dtype
            assert isinstance(node, OpNode)
            node.num_outputs = len(checked_type.fields)
            return tuple(ret)
        # Normal tensor return type
        if not isinstance(checked_type, TensorType):
            raise RuntimeError("type %s not supported" % checked_type)
        '''print(node.name)
        if node.op_name=='reshape':
            node.attrs["shape"] = node.op_attrs['newshape']
            import ipdb
            ipdb.set_trace()'''
        node.attrs["shape"] = [shape_to_json(checked_type.shape)]
        node.attrs["dtype"] = [checked_type.dtype]
        node.num_outputs = 1
        node.attrs["id"] = node_id
        #print(node)
        self.idNodeMap[node_id] = node
        if 'inputs' in dir(node): 
            for item in node.inputs:
                self.idOutputMap[item].append(node_id)
                #print(item,':',self.nodeOpDict[item])
                if 'op_name' in dir(node):
                    if node.op_name in ['add','multiply'] and self.nodeOpDict[item] in ['add','multiply','nn.leaky_relu','nn.relu']:
                        self.fusePairs[item] = node_id
                        #print('[INFO] Need Fusing')
        if 'op_name' in dir(node):
            self.nodeOpDict[node_id] = node.op_name
        elif 'input' in node.name: 
            self.nodeOpDict[node_id] = 'input'
        else:
            self.nodeOpDict[node_id] = 'const'
        return NodeRef(node_id, 0)
        
    def _get_unique_name(self, name):
        if name not in self._name_map:
            self._name_map[name] = 1
            return name
        index = self._name_map[name]
        self._name_map[name] += 1
        return self._get_unique_name(name + str(index))