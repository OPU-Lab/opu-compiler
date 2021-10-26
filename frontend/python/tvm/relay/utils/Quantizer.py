from .. import transform as _transform
from .. import expr
from ..module import Module as _Module
from ..build_module import BuildModule,_update_target
from tvm import relay
from ... import autotvm
from . import tvmFuncParser 
from . import sim
class Quantizer:
    def __init__(self, mod, params, config_dict):
        with relay.build_config(opt_level=3):
            func = self.optimize(mod, params)
        parser = tvmFuncParser.Parser(func)  
        parser.collectNodes()
        #TODO: merge conv-add(bias)-mul(bn)-add(bn) to conv-add
        parser.check()
        #import ipdb
        #ipdb.set_trace()
        input_shape = config_dict['input_shape']
        fx_output_dir = config_dict['output_dir']
        simulator = sim.simGen(parser.nodes, parser.params, input_shape, fx_output_dir, config_dict)
        self.sim_result = simulator.quantize_with_tf_sim()
    
    def get_output(self):
        return self.sim_result
    
    def optimize(self, mod, params):  
        #import ipdb
        #ipdb.set_trace()
        '''if params:
            mod['main'] = self._bind_params(mod['main'], params)
        _optimize = _transform.Sequential([_transform.SimplifyInference(),
                                      _transform.FoldConstant(),
                                      _transform.FoldScaleAxis(),
                                      _transform.CanonicalizeOps(),
                                      _transform.FoldConstant()])#,
                                      #_transform.FuseOps(0)])
        mod = _optimize(mod)
        import ipdb
        ipdb.set_trace()
        return mod['main']'''
        if isinstance(mod, _Module):
            func = mod["main"]
        elif isinstance(mod, expr.Function):
            func = mod
            print(
                "Please use input parameter mod (tvm.relay.module.Module) "
                "instead of deprecated parameter func (tvm.relay.expr.Function)")
        else:
            raise ValueError("Type of input parameter mod must be tvm.relay.module.Module")
        target = _update_target('llvm')
        if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
            tophub_context = autotvm.tophub.context(list(target.values()))
        else:
            tophub_context = autotvm.util.EmptyContext()

        with tophub_context:
            bld_mod = BuildModule()
            mod, params = bld_mod.optimize(func, target, params)
        return mod['main']
    
    def _bind_params(self, func, params):
        """Bind the params to the expression.
        """
        name_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in name_dict:
                name_dict[name] = None
            else:
                name_dict[name] = arg
        bind_dict = {}
        for k, v in params.items():
            if k not in name_dict:
                continue
            arg = name_dict[k]
            if arg is None:
                raise ValueError("Multiple args in the function have name %s" % k)
            bind_dict[arg] = expr.const(v)
        return expr.bind(func, bind_dict)