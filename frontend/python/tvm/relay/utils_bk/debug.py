import numpy as np
import tensorflow as tf
from . import suppress_stdout_stderr as block
import copy 

class Debugger:
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = './'
        self.enable = False
        self.opu_cfg_template = {
                                    'FM':None,
                                    'WEIGHT':None,
                                    'BIAS':None,
                                    'IPA':None,
                                    'RESIDUAL':None,
                                    'POOLING':None,
                                    'ACTIVATION':None
                                }
        self.cfg_dict = {}
        
        
    def opu_layout_for_input(self, ifm, weight, fracLen, feed_dict):
        ifm_shape = ifm.shape
        if not isinstance(ifm_shape[0], int):
            ifm_shape = [x.value for x in ifm_shape]
            with block.suppress_stdout_stderr():
                with tf.Session() as sess:
                    ifm = sess.run(ifm, feed_dict)
        kernel_shape = weight.shape
        if not isinstance(kernel_shape[0], int):
            kernel_shape = [x.value for x in kernel_shape]
        Kh,Kw,Cin = kernel_shape[-3:]
        H,W,Cin = ifm_shape[-3:]
        H_nopad = H-Kh+1
        W_nopad = W-Kw+1
        dstImg = np.array([H_nopad, W_nopad, 64])
        for h in range(H_nopad):
            for w in range(W_nopad):
                cnt = 0
                for c in range(Cin):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            dstImg[h][w][32+cnt] = ifm[0][h+kh][w+kw][c]
                            cnt+=1
                            
    def check_type(self, data, feed_dict=None):
        if not isinstance(data, np.ndarray):
            with block.suppress_stdout_stderr():
                with tf.Session() as sess:
                    if feed_dict is not None:
                        data = sess.run(data, feed_dict)
                    else:
                        data = sess.run(data)
        return data
    
    def get_cfg(self, layerId):
        if layerId in self.cfg_dict.keys():
            return self.cfg_dict[layerId]
        else:
            cfg = copy.deepcopy(self.opu_cfg_template)
            self.cfg_dict[layerId] = cfg
            return cfg
    
    def collect(self, layerId, name, data, feed_dict=None):    
        if not self.enable: return
        data = self.check_type(data, feed_dict)
        cfg = self.get_cfg(layerId)
        cfg[name] = data
        print('[DEBUG]',name,data.shape,'collected for layer',layerId)
        if name=='POOLING':
            
            import ipdb
            ipdb.set_trace()
            print()
        