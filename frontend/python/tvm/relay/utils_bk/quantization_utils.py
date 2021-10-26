import numpy as np
import tensorflow as tf
import math
from .cython.frexp import PyCppClass
from . import suppress_stdout_stderr as block
def tf_float2fx_floor(value, wl, fl):
    gap = tf.constant(2.0**(-fl))
    minVal = tf.constant(-2.0**(wl-1))*gap
    maxSteps = 2.0**wl
    nSteps = tf.clip_by_value(tf.math.floordiv((value-minVal),gap),0.0,maxSteps)
    val_fp = minVal+nSteps*gap
    return val_fp
    
def tf_float2fx_round(value, wl, fl):
    gap = tf.constant(2.0**(-fl))
    minVal = tf.constant(-2.0**(wl-1))*gap
    maxSteps = 2.0**wl
    nSteps = tf.clip_by_value(tf.round(tf.divide((value-minVal),gap)),0.0, maxSteps)
    val_fp = minVal+nSteps*gap
    return val_fp
# Functionalities for users to exploit: 
# granularity: whole; channel-wise;
# dtype: scale; pow2-fixedpoint; floatingpoint(ieee standard) 
# local measurement: mse; kl-divergence
# search policy: naive-traversal
class Qnn:
    def __init__(self):
        pass
    
    # dtype convertion: basic functions           
    def to_fixedpoint(self, data_i, word_len, frac_len, type='round'):
        if type=='floor':
            return tf_float2fx_floor(data_i, word_len, frac_len)
        elif type=='round':
            return tf_float2fx_round(data_i, word_len, frac_len)
        else:
            assert 0, '[ABORT] Unrecognized fixedpoint type:'+type
    
    def to_floatpoint(self, data_i, word_len, exp_len):
        pass
    
    def to_scaled_int(self):
        pass
        
    def convert(self, data_i, word_len, frac_len, symbolic=False):
        if symbolic is True:
            data_q = self.to_fixedpoint(data_i, word_len, frac_len)
        else:
            with tf.Graph().as_default():
                data_q = self.to_fixedpoint(data_i, word_len, frac_len)
                with block.suppress_stdout_stderr():
                    with tf.Session() as sess:
                        data_q = sess.run(data_q)
        return data_q    
        
    # error measurement
    def deviation(self, data_q, data_origin):    
        return self.mse(data_q, data_origin)
        
    def mse(self, x, y):
        x = x.flatten()
        y = y.flatten()
        dif = np.sum((x-y)**2)/len(y)
        return dif
    
    # search policy
    def search(self, data_i, word_len):
        fl_init = -8
        fl = fl_init
        errs = []
        while fl < 20:
            data_q = self.convert(data_i, word_len, fl)
            err = self.deviation(data_q, data_i)
            #print('\tfl:',fl,' dif:',err)
            if len(errs)>0 and err>errs[-1] and fl>0:
                least_err_idx = int(np.argmin(np.array(errs)))
                return least_err_idx+fl_init
            errs.append(err)
            fl+=1
        return fl-1
    
    # granularity
    def apply(self, data_i, word_len):
        fl_opt = self.search(data_i, word_len)
        data_q = self.convert(data_i, word_len, fl_opt)
        return data_q, fl_opt
    
    