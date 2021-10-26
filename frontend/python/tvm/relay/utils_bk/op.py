import numpy as np
import time
import threading
import tensorflow as tf
#from . import fp as fixpoint
from . import suppress_stdout_stderr as block
import scipy.io
import os
from .quantization_utils import Qnn
from .global_var import global_var as gv
global_var = gv()
qnn = Qnn()
def dump2file(filename, string):
    with open(filename, 'a') as f:
        print(string, file=f)
    name = string.split('=')[0].split('_')[-1]
    fl = string.split('=')[1]
    if name=='fm':
        global_var.fracLenDict['fm'].append(int(fl))
    elif name=='weight':
        global_var.fracLenDict['weight'].append(int(fl))
    elif name=='bias':
        global_var.fracLenDict['bias'].append(int(fl))

def dimension_check(target, ifm):
    out = ifm
    ndim = len(out.shape)
    if ndim<2:
        out = tf.expand_dims(out, 0)
    if target=="opu" and not isinstance(out, np.ndarray):
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    
        
def conv2d_transpose(target, ifm, weight, output_shape, stride, padding):
    '''if target=='opu':
        weight_fp = fixpoint.fp(weight, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(weight, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)'''
    out = tf.nn.conv2d_transpose(ifm, filter=weight, output_shape=output_shape, strides=[1,stride, stride,1], padding='SAME')
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out  

def conv3d_transpose(target, ifm, weight, output_shape, stride, padding):
    '''if target=='hw':
        weight_fp = fixpoint.fp(weight, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(weight, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)'''
    out = tf.nn.conv3d_transpose(ifm, filter=weight, output_shape=output_shape, strides=[1,stride, stride,stride,1], padding='SAME')
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    

def conv3d(target, ifm, weight, strides, padding, platform):
    '''if target=='opu':
        weight_fp = fixpoint.fp(weight, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(weight, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)'''
    if padding=='SAME' and platform is not None:
        kz = weight.shape
        if not isinstance(kz[0],int):
            kz = [x.value for x in kz]
        ifm_shape = ifm.shape
        if not isinstance(ifm_shape[0],int):
            ifm_shape = [x.value for x in ifm_shape]
        D,H,W = ifm_shape[1:4]
        Kd,Kh,Kw = kz[0:3]
        Sd,Sh,Sw = strides
        Pd = (np.ceil(D/Sd)-1)*Sd+Kd-D
        Pd_r = int(Pd//2)
        Pd_l = int(Pd-Pd_r)
        Ph = (np.ceil(H/Sh)-1)*Sh+Kh-H
        Ph_r = int(Ph//2)
        Ph_l = int(Ph-Ph_r)
        Pw = (np.ceil(W/Sw)-1)*Sw+Kw-W
        Pw_r = int(Pw//2)
        Pw_l = int(Pw-Pw_r)
        paddings = tf.constant([[0,0],[Pd_l,Pd_r],[Ph_l,Ph_r],[Pw_l,Pw_r],[0,0]])
        ifm = tf.pad(ifm, paddings, 'CONSTANT')
        out = tf.nn.conv3d(ifm, filter=weight, strides=[1,strides[0], strides[1], strides[2] ,1], padding='VALID')
    else:
        out = tf.nn.conv3d(ifm, filter=weight, strides=[1,strides[0], strides[1], strides[2] ,1], padding=padding)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def max_pool3d(target, ifm, kz, strides, padding, platform):      
    if padding=='SAME' and platform is not None:
        if not isinstance(kz[0],int):kz = [x.value for x in kz]
        ifm_shape = ifm.shape
        if not isinstance(ifm_shape[0],int):ifm_shape = [x.value for x in ifm_shape]
        D,H,W = ifm_shape[1:4]
        Kd,Kh,Kw = kz[0:3]
        Sd,Sh,Sw = strides
        Pd = (np.ceil(D/Sd)-1)*Sd+Kd-D
        Pd_r = int(Pd//2)
        Pd_l = int(Pd-Pd_r)
        Ph = (np.ceil(H/Sh)-1)*Sh+Kh-H
        Ph_r = int(Ph//2)
        Ph_l = int(Ph-Ph_r)
        Pw = (np.ceil(W/Sw)-1)*Sw+Kw-W
        Pw_r = int(Pw//2)
        Pw_l = int(Pw-Pw_r)
        pre_padding = tf.constant([[0,0],[Pd_l,Pd_r],[Ph_l,Ph_r],[Pw_l,Pw_r],[0,0]])
        ifm = tf.pad(ifm, pre_padding, 'CONSTANT')
        padding = 'VALID'
    out = tf.nn.max_pool3d(ifm, [1,kz[0],kz[1],kz[2],1], [1,strides[0], strides[1], strides[2] ,1], padding)# data_format='NDHWC'
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    
    
def avg_pool3d(target, ifm, kz, strides):      
    out = tf.nn.avg_pool3d(ifm, [1,kz[0],kz[1],kz[2],1], [1,strides[0], strides[1], strides[2] ,1], padding='VALID')
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    
    
def image_resize_with_zero_tensor(ifm, size):
    tshape = [ifm.shape[0], size[0], size[1], ifm.shape[3]]
    zeros = tf.zeros(tshape, tf.float32)
    
    print()

def image_resize(target, ifm, size, method):
    method_dict = {'BILINEAR':0,'NEAREST_NEIGHBOR':1,'BICUBIC':2,'AREA':3}
    out = tf.image.resize_images(images=ifm, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#method_dict[method])
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out        
    
def strided_slice(target, ifm, begin, end, axis=3):
    out = tf.strided_slice(ifm, begin, end)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out        
        
def upsampling(target, ifm, scale, method='nearest'):
    #import ipdb
    #ipdb.set_trace()
    if ifm.shape[1]==ifm.shape[2]:
        size = [x*scale for x in ifm.shape[1:3]]
        out = tf.image.resize_images(ifm, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)
        #out = tf.transpose(out,perm=[0,3,1,2])
    else:
        size = [x*scale for x in ifm.shape[2:]]
        ifm = tf.transpose(ifm,perm=[0,2,3,1])
        out = tf.image.resize_images(ifm, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)
        #out = tf.transpose(out,perm=[0,3,1,2])
    # zero-insertion    
    #import ipdb
    #ipdb.set_trace()
    '''if target=='opu':
        if not isinstance(ifm, np.ndarray):
            with tf.Session() as sess:
                ifm = sess.run(ifm)
        size = [x*scale for x in ifm.shape[1:3]]
        shape = [ifm.shape[0], size[0], size[1], ifm.shape[3]]
        if not isinstance(ifm.shape[0],int):
            ishape = [x.value for x in ifm.shape]
        else:
            ishape = ifm.shape
        if not isinstance(shape[0],int):
            shape = [x.value for x in shape]
        tmp = np.zeros(shape)
        for n in range(ishape[0]):
            for i in range(ishape[1]):
                for j in range(ishape[2]):
                    for c in range(ishape[3]):
                        tmp[n][i*2][j*2][c] = ifm[n][i][j][c]
        out = tmp.astype(np.float32)
        out = tf.constant(out)'''
    if target=='opu':
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def dense(target, ifm, w, layerId, basedir):
    ishape = ifm.shape
    if len(ishape)==2:
        ifm = ifm
    elif ishape[1]==1 and ishape[2]==1:
        ifm = tf.squeeze(ifm, axis=[1,2])
    else:
        ifm = tf.squeeze(ifm)#, axis=[1,2])
    ifm = dimension_check('sw',ifm)    
    out = tf.matmul(tf.cast(ifm,tf.float32),w.transpose(1,0))
    '''if target=='opu':
        weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)
        scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
        dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
        if not weight.dtype==ifm.dtype:
            weight = weight.astype(ifm.dtype)
        out = tf.matmul(ifm,weight.transpose(1,0))
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
        out = out.astype(np.float32)'''
    return out
        
def reshape(target, ifm, out_shape, platform):
    if platform is not None:
        ifm_shape = ifm.shape
        if not isinstance(ifm_shape[0], int):ifm_shape = [x.value for x in ifm_shape]
        dims = len(ifm.get_shape())
        if dims==4:
            pt_perm = [0,dims-1]
            for item in range(1,dims-1):
                pt_perm.append(item)
            ifm = tf.transpose(ifm,perm=pt_perm)
            print('transpose',pt_perm)
    out = tf.reshape(ifm,out_shape)
    if target=='opu':
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
    
def yolo_reorg(target, ifm, out_shape, stride=2):
    channels = out_shape[-1]//(stride*stride)
    _height, _width, _channel = out_shape[1], out_shape[2], out_shape[3]
    net = tf.reshape(ifm, [-1, _height, stride, _width, stride, channels])
    net = tf.transpose(net, [0, 1, 3, 2, 4, 5]) # batch_size, _height, _width, stride, stride, channels
    out = tf.reshape(net, out_shape)
    if target=='opu':
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    
    

def mean(target, ifm, out_shape):
    mean_dims = []
    for i in range(1,len(ifm.shape)+1):
        if not ifm.shape[i]==out_shape[i]:
            mean_dims.append(i)
    out = tf.reduce_mean(ifm, mean_dims, keepdims=True)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def residualAdd(target, operand_0, operand_1):
    out = tf.add(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
        
def multiply(target, operand_0, operand_1):
    out = tf.multiply(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def divide(target, operand_0, operand_1):
    out = tf.div(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out    
    
def subtract(target, operand_0, operand_1):
    out = tf.subtract(operand_0, operand_1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
    
def transpose(target, ifm, axes, out_shape):
    if isinstance(ifm, np.ndarray):
        ifm_shape = ifm.shape
    else:
        ifm_shape = [x.value for x in ifm.shape]
    if tuple(ifm_shape)==tuple(out_shape):
        return ifm   
    out = tf.transpose(ifm, perm=axes)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    assert out.shape==tuple(out_shape)
    return out

def tfpad(target, ifm, pad_width):
    paddings = tf.constant([[pad_width[i][0].value, pad_width[i][1].value] for i in range(len(pad_width))])
    ofm = tf.pad(ifm, paddings, 'CONSTANT')
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                ofm = sess.run(ofm)
    return ofm

def relu(target, ifm):
    out = tf.nn.relu(ifm)
    if target=="opu":
        out = tf.nn.relu(ifm)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
    return out

def tanh(target, ifm):
    if target=="sw":
        return tf.nn.tanh(ifm)
    elif target=="opu":
        out = tf.nn.tanh(ifm)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
        return out        

def sigmoid(target, ifm):
    out = tf.nn.sigmoid(ifm)
    if target=="opu":
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
    return out
        
def clip(target, ifm, min, max):
    out = tf.clip_by_value(ifm, min, max)
    if target=="opu":
        out = tf.clip_by_value(ifm, min, max)
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
    return out        
        
        
def concat(target, ifms, axis):
    out = tf.concat(ifms, axis)
    if target=="opu":
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
    return out

def conv2d(target, ifm, weight, strides, padding, data_format, kernel_format, groups, inpFracLen, layerId, cutposLen, basedir, feed_dict=None, platform=None):
    '''if data_format=='NCHW':# ->NHWC
        if target=='opu':
            ifm = ifm.transpose(0,2,3,1)
            #weight = weight.transpose(0,2,3,1)
        elif target=='sw':
            ifm = tf.transpose(ifm, perm=[0,2,3,1])
            #weight = tf.transpose(weight, perm=[0,2,3,1])
    if kernel_format=='OIHW':# ->HWIO
        if target=='opu':
            weight = weight.transpose(2,3,1,0)
        elif target=='sw':
            weight = tf.transpose(weight, perm=[2,3,1,0])
    elif kernel_format=='OHWI':
        if target=='opu':
            weight = weight.transpose(1,2,3,0)
        elif target=='sw':
            weight = tf.transpose(weight, perm=[1,2,3,0])
    elif kernel_format=='HWOI':
        if target=='opu':
            weight = weight.transpose(0,1,3,2)
        elif target=='sw':
            weight = tf.transpose(weight, perm=[0,1,3,2])'''
    # target = 'sw'
    if groups is not None and groups>1:
        if isinstance(ifm.shape[0], int):
            in_channels = ifm.shape[-1]
        else:
            in_channels = ifm.shape[-1].value
        if groups==in_channels:
            return conv2d_depthwise(target, ifm, weight, strides, padding, 'NHWC', groups, inpFracLen, layerId, cutposLen, basedir)
        else: # group conv - tf.split tf.concat
            return conv2d_group(target, ifm, weight, strides, padding, 'NHWC', groups, inpFracLen, layerId, cutposLen, basedir)
    if target=="opu":
        out = conv2d_opu(ifm, weight, strides[1], padding, inpFracLen, layerId, cutposLen, basedir)
    elif target=="sw":
        if padding=='SAME' and platform is not None:
            kz = weight.shape
            if not isinstance(kz[0],int):
                kz = [x.value for x in kz]
            ifm_shape = ifm.shape
            if not isinstance(ifm_shape[0],int):
                ifm_shape = [x.value for x in ifm_shape]
            H,W = ifm_shape[1:3]
            Kh,Kw = kz[0:2]
            Sh,Sw = strides[1:3]
            Ph = max(0,(np.ceil(H/Sh)-1)*Sh+Kh-H)
            Ph_r = int(Ph//2)
            Ph_l = int(Ph-Ph_r)
            Pw = max(0,(np.ceil(W/Sw)-1)*Sw+Kw-W)
            Pw_r = int(Pw//2)
            Pw_l = int(Pw-Pw_r)
            paddings = tf.constant([[0,0],[Ph_l,Ph_r],[Pw_l,Pw_r],[0,0]])
            ifm = tf.pad(ifm, paddings, 'CONSTANT')
            out = tf.nn.conv2d(ifm, weight, strides=strides, padding='VALID')
        else:
            out = tf.nn.conv2d(ifm, weight, strides=strides, padding=padding)
    elif target=='hw': 
        out = tf.nn.conv2d(ifm, weight, strides=strides, padding=padding)
        # out = conv2d_opu(ifm, weight, strides, padding, feed_dict, cutposLen, 16)
        '''ref_i = ifm        
        weight_t = weight#ref_w.transpose(2,3,1,0)#tf.convert_to_tensor(ref_w.transpose(2,3,1,0),dtype=tf.float32)
        if padding=='SAME':
            kz = weight.shape[0].value
            if kz == 1:
                pz = 0
            else:
                pz = 1
            pre_padding = tf.constant([[0,0],[pz,pz],[pz,pz],[0,0]])
            ref_i = tf.pad(ref_i, pre_padding, 'CONSTANT')
            padding = 'VALID'
        out = tf.nn.conv2d(ref_i, weight_t, strides=strides, padding=padding)'''
        
    else:
        assert 0,"unknown target"
    '''if data_format=='NCHW':
        if target=='opu':
            out = out.transpose(0,3,1,2)
        elif target=='sw':
            out = tf.transpose(out,perm=[0,3,1,2])'''
    return out
    
def conv2d_opu_io_par(ifm, weight, strides, padding, feed_dict, intermediate_length=16):
    '''identify whether padding is needed according to ifm size & kernel size & strides & padding'''
    ifm_size = ifm.shape[1:] #NHWC
    if not isinstance(ifm_size[0],int): ifm_size = [x.value for x in ifm_size]
    ker_size = weight.shape #HWIO
    if not isinstance(ker_size[0],int): ker_size = [x.value for x in ker_size]
    if padding=='SAME':
        ofm_size = [ifm_size[i]/strides[i+1] for i in range(3)]
        pad_h, pad_w = [int(max(strides[1]*(ofm_size[0]-1)+ker_size[0]-ifm_size[0],0)),int(max(strides[2]*(ofm_size[1]-1)+ker_size[1]-ifm_size[1],0))]
        if sum([pad_h, pad_w])>0:
            pad_h_0 = pad_h//2
            pad_h_1 = pad_h - pad_h_0
            pad_w_0 = pad_w//2
            pad_w_1 = pad_w - pad_w_0
            paddings = tf.constant([[0,0],[pad_h_0,pad_h_1],[pad_w_0,pad_w_1],[0,0]])
            ifm = tf.pad(ifm, paddings, 'CONSTANT')            
            ifm_size = ifm.shape[1:] #NHWC
            if not isinstance(ifm_size[0],int): ifm_size = [x.value for x in ifm_size]
    '''with block.suppress_stdout_stderr():
        fl_local = []
        with tf.Session() as sess:
            for kx in range(ker_size[0]):
                for ky in range(ker_size[1]):
                    tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=strides,padding=padding)
                    tmp = sess.run(tmp, feed_dict)
                    #fl_t = frange(intermediate_length, tmp.flatten())
                    fl_t = fixpoint.fp(tmp, wordLen=intermediate_length, opt=True)._fl
                    fl_local.append(fl_t)
    cutposLen = min(fl_local) '''
    #print('cutposLen =',cutposLen)
    #print(fl_local)   
    
    n = 64
    ci_par = [[i,min(i+n,ker_size[2])] for i in range(0, ker_size[2], n)]   
    print(ci_par)
    for ci,item_ci in enumerate(ci_par):
        cin_st,cin_et = item_ci
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                tmp = tf.nn.conv2d( ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,cin_st:cin_et],
                                    weight[kx:kx+1,ky:ky+1,cin_st:cin_et,:],strides=strides,padding=padding)
                tmp = tf.cast(tmp, tf.float32)
                if kx==0 and ky==0 and ci==0:
                    ofm = tmp
                else:      
                    ofm += tmp 
                    ofm = tf.cast(ofm, tf.float32)
                    #ofm = tf_float2fx_floor(ofm, intermediate_length, cutposLen)
    return ofm
    
    
def conv2d_opu(ifm, weight, strides, padding, feed_dict, cutposLen, intermediate_length=16):
    '''identify whether padding is needed according to ifm size & kernel size & strides & padding'''
    ifm_size = ifm.shape[1:] #NHWC
    if not isinstance(ifm_size[0],int): ifm_size = [x.value for x in ifm_size]
    ker_size = weight.shape #HWIO
    if not isinstance(ker_size[0],int): ker_size = [x.value for x in ker_size]
    if padding=='SAME':
        ofm_size = [ifm_size[i]/strides[i+1] for i in range(3)]
        pad_h, pad_w = [int(max(strides[1]*(ofm_size[0]-1)+ker_size[0]-ifm_size[0],0)),int(max(strides[2]*(ofm_size[1]-1)+ker_size[1]-ifm_size[1],0))]
        if sum([pad_h, pad_w])>0:
            pad_h_0 = pad_h//2
            pad_h_1 = pad_h - pad_h_0
            pad_w_0 = pad_w//2
            pad_w_1 = pad_w - pad_w_0
            paddings = tf.constant([[0,0],[pad_h_0,pad_h_1],[pad_w_0,pad_w_1],[0,0]])
            ifm = tf.pad(ifm, paddings, 'CONSTANT')            
            ifm_size = ifm.shape[1:] #NHWC
            if not isinstance(ifm_size[0],int): ifm_size = [x.value for x in ifm_size]
    with block.suppress_stdout_stderr():
        fl_local = []
        with tf.Session() as sess:
            for kx in range(ker_size[0]):
                for ky in range(ker_size[1]):
                    tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=strides,padding=padding)
                    tmp = sess.run(tmp, feed_dict)
                    #fl_t = frange(intermediate_length, tmp.flatten())
                    #fl_t = fixpoint.fp(tmp, wordLen=intermediate_length, opt=True)._fl
                    #_, fl_t = qnn.apply(tmp, intermediate_length)
                    fl_t = qnn.search(tmp, intermediate_length)
                    fl_local.append(fl_t)
    cutposLen = min(fl_local)       
    print('cutposLen =',cutposLen)
    print(fl_local)
    for kx in range(ker_size[0]):
        for ky in range(ker_size[1]):
            tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=strides,padding=padding)
            if kx==0 and ky==0:
                ofm = tmp
            else:
                ofm += tmp
                #cutposLen = fl_local[kx*ker_size[1]+ky]
                ofm = tf_float2fx_floor(ofm, intermediate_length, cutposLen)
    return ofm
    
    
def conv2d_group(target, ifm, weight, strides, padding, data_format, groups, inpFracLen, layerId, cutposLen, basedir):
    '''if target=='opu':
        weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
        weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
        weight = weight_fp._d_fp
        print('<>kernel_fracLen=',weight_fp._fl)
        scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
        dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))'''
    sub_ifms = tf.split(ifm, groups, axis=3)#NHWC
    sub_weights = tf.split(weight, groups, axis=3)#HWIO
    sub_outs = []
    for i in range(groups):
        tmp = tf.nn.conv2d(sub_ifms[i], sub_weights[i], strides=strides, padding=padding)
        sub_outs.append(tmp)
    out = tf.concat(sub_outs,axis=3)
    out = tf.identity(out)
    if target=='opu':
        with block.suppress_stdout_stderr():
             with tf.Session() as sess:
                 out = sess.run(out)
    return out    
    
    
def conv2d_depthwise(target, ifm, weight, strides, padding, data_format, groups, inpFracLen, layerId, cutposLen, basedir):
    if target=='sw':
        weight = tf.transpose(weight, perm=[0,1,3,2])
        out = tf.nn.depthwise_conv2d(input=ifm, filter=weight, strides=strides, padding=padding, data_format=data_format)
    elif target == 'hw':    
        weight = tf.transpose(weight, perm=[0,1,3,2])
        out = tf.nn.depthwise_conv2d(input=ifm, filter=weight, strides=strides, padding=padding, data_format=data_format)   
    elif target=='opu':
        weight = weight.transpose(0,1,3,2)
        out = conv2d_depthwise_opu(ifm, weight, strides[1], padding, inpFracLen, layerId, cutposLen, basedir)
    else:
        assert 0,"unknown target"
    return out

def conv2d_depthwise_opu(ifm, w, stride, padding, inpFracLen, layerId, cutposLen, basedir):
    #weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
    #weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
    weight = w #weight_fp._d_fp
    #print('<>kernel_fracLen=',weight_fp._fl)
    #scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
    #dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
    #prod_fracLen = weight_fp._fl+inpFracLen
    cutposLen = 15 - (1+7-cutposLen) 
    #print('<>*',prod_fracLen,'->',cutposLen)
    #
    global_var.fracLenDict['cutposLen'].append(cutposLen)      
    fm_size = [int(ifm.shape[1]),int(ifm.shape[2])]
    depth = 1
    ker_size = [int(w.shape[0]),int(w.shape[1])]
    ker_num = int(w.shape[2]) # different from conv2d
    if padding=='SAME':
        ofm_size = [x//stride for x in fm_size]
    else:
        temp = [fm_size[0]-ker_size[0]+1,fm_size[1]-ker_size[1]+1]
        ofm_size = [(x+stride-1)//stride for x in temp]
    ofm = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                tmp = tf.nn.depthwise_conv2d(ifm[:,kx:int(fm_size[0])-ker_size[0]+kx+1,ky:int(fm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                ofm+=tmp
                #ofm = fixpoint.fp(ofm,wordLen=16,fracLen=cutposLen, roundMethod='floor')._d_fp
                ofm, _ = qnn.convert(ofm, 16, cutposLen)
    return ofm
    
    
def biasAdd(target, ifm, bias, layerId, basedir):
    out = tf.nn.bias_add(ifm, bias)
    if target=="opu":
        out = biasAdd_opu(ifm, bias, layerId, basedir) 
    return out
        

def leakyRelu(target, ifm, alpha):
    out = tf.nn.leaky_relu(ifm, alpha)
    if target=="opu":
        out = tf.nn.leaky_relu(ifm, alpha)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def maxPool(target, ifm, kz, strides, pad_mode, data_format, platform, pad):
    if pad_mode=='SAME' and platform is not None:
        if not isinstance(kz[0],int):
            kz = [x.value for x in kz]
        ifm_shape = ifm.shape
        if not isinstance(ifm_shape[0],int):
            ifm_shape = [x.value for x in ifm_shape]
        H,W = ifm_shape[1:3]
        Kh,Kw = kz[1:3]
        Sh,Sw = strides[1:3]
        Ph = max(0,(np.ceil(H/Sh)-1)*Sh+Kh-H)
        Ph_r = int(Ph//2)
        Ph_l = int(Ph-Ph_r)
        Pw = max(0,(np.ceil(W/Sw)-1)*Sw+Kw-W)
        Pw_r = int(Pw//2)
        Pw_l = int(Pw-Pw_r)
        pad = [x.value for x in pad]
        if not [Ph_l,Pw_l,Ph_r,Pw_r]==pad: # squeezenet 111x111 pad[0,0,1,1] stride=2,kz=3x3->55x55
            Ph_l,Pw_l,Ph_r,Pw_r = pad
        paddings = tf.constant([[0,0],[Ph_l,Ph_r],[Pw_l,Pw_r],[0,0]])
        ifm = tf.pad(ifm, paddings, 'CONSTANT')
        out = tf.nn.max_pool2d(ifm, ksize=kz, strides=strides, padding='VALID', data_format=data_format)
    else:
        out = tf.nn.max_pool2d(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
    if target=="opu":
        out = tf.nn.max_pool2d(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def avgPool(target, ifm, kz, strides, pad_mode, data_format):
    out = tf.nn.avg_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
    if target=="opu":
        out = tf.nn.avg_pool(ifm, ksize=kz, strides=strides, padding=pad_mode, data_format=data_format)
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out

def globalAvgPool(target, ifm):
    out = tf.reduce_mean(ifm, axis=[1,2])
    out = tf.expand_dims(out,1)
    out = tf.expand_dims(out,1)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out
 
def expand_dims(target, ifm, axis, num_newaxis):
    '''out = ifm
    for i in range(num_newaxis):
        out = tf.expand_dims(out,axis)
    ndim = len(out.shape)
    for i in range(4-ndim):
        out = tf.expand_dims(out,0)
    out = tf.transpose(out, perm=[0,2,3,1])'''
    out = ifm
    ndim = len(out.shape)
    for i in range(4-ndim):
        out = tf.expand_dims(out,0)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out 
    
def squeeze(target, ifm, axis):
    out = tf.squeeze(ifm, axis)
    if target=="opu":
        with block.suppress_stdout_stderr():
            with tf.Session() as sess:
                out = sess.run(out)
    return out 

def biasAdd_opu(fm, bias, layerId, basedir):
    '''bias_fp = fixpoint.fp(bias,wordLen=16,opt=True)
    bias_fp = fixpoint.fp(bias,wordLen=16,fracLen=bias_fp._fl)
    b0 = bias
    bias = bias_fp._d_fp
    print('<>bias_fracLen=',bias_fp._fl)'''
    #import ipdb
    #ipdb.set_trace()
    #scipy.io.savemat(basedir+'bias_'+str(layerId)+'.mat',{'value':bias})
    #dump2file(basedir+'fracLen.txt',str(layerId)+'_bias='+str(bias_fp._fl))
    out = tf.nn.bias_add(fm, bias)
    with block.suppress_stdout_stderr():
        with tf.Session() as sess:
            out = sess.run(out)
    #out = fixpoint.fp(out,wordLen=16,opt=True)._d_fp
    return out
    

def conv2d_opu_0(fm, w, stride, padding, inpFracLen, layerId, cutposLen, basedir):
    assert fm.shape[3] == w.shape[2]
    fm_size = [int(fm.shape[1]),int(fm.shape[2])]
    depth = int(fm.shape[3])
    ker_size = [int(w.shape[0]),int(w.shape[1])]
    ker_num = int(w.shape[3])
    if padding=='SAME':
        ofm_size = [x//stride for x in fm_size]
    else:
        temp = [fm_size[0]-ker_size[0]+1,fm_size[1]-ker_size[1]+1]
        ofm_size = [(x+stride-1)//stride for x in temp]
    # compute padding size
    pad_num = [int(max(stride*(ofm_size[0]-1)+ker_size[0]-fm_size[0],0)),int(max(stride*(ofm_size[1]-1)+ker_size[1]-fm_size[1],0))]
    pad_size_0 = [int(x//2) for x in pad_num]
    pad_size_1 = [int(pad_num[0] - pad_size_0[0]),int(pad_num[1] - pad_size_0[1])]
    # padding 0s
    ifm_size = [int(pad_num[0]+fm_size[0]),int(pad_num[1]+fm_size[1])]
    ifm = np.ndarray([1,ifm_size[0],ifm_size[1],depth])
    for d in range(depth):
        for i in range(ifm_size[0]):
            for j in range(pad_size_0[1]):
                ifm[0][i][j][d] = 0
        for i in range(pad_size_0[0]):
            for j in range(ifm_size[1]):
                ifm[0][i][j][d] = 0
        for i in range(fm_size[0]):
            for j in range(fm_size[1]):
                ifm[0][pad_size_0[0]+i][pad_size_0[1]+j][d]=fm[0][i][j][d]
        for i in range(pad_size_0[0]+fm_size[0], ifm_size[0]):
            for j in range(ifm_size[1]):
                ifm[0][i][j][d] = 0
        for i in range(ifm_size[0]):
            for j in range(pad_size_0[1]+fm_size[1], ifm_size[1]):
                ifm[0][i][j][d] = 0
    ofm = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    ofm0 = np.zeros([1,int(ofm_size[0]), int(ofm_size[1]), ker_num])
    #weight_fp = fixpoint.fp(w, wordLen=8,opt=True)
    #weight_fp = fixpoint.fp(w, fracLen=weight_fp._fl)
    weight = w #weight_fp._d_fp
    #print('<>kernel_fracLen=',weight_fp._fl)
    #import ipdb
    #ipdb.set_trace()
    scipy.io.savemat(basedir+'weight_'+str(layerId)+'.mat',{'value':weight})
    #dump2file(basedir+'fracLen.txt',str(layerId)+'_weight='+str(weight_fp._fl))
    #prod_fracLen = weight_fp._fl+inpFracLen
    cutposLen = 15 - (1+7-cutposLen) 
    #print('<>*',prod_fracLen,'->',cutposLen)
    #cutposLen = min(prod_fracLen, cutposLen)
    fl_local = [cutposLen]
    
    '''global_var.fracLenDict['cutposLen'].append(cutposLen) 
    ofm = tf.nn.conv2d(ifm, weight, strides=[1,stride,stride,1], padding=padding)
    with tf.Session() as sess:
        ofm = sess.run(ofm)
    return ofm'''
    
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                fl_t = frange(16, tmp.flatten())
                fl_local.append(fl_t)
    cutposLen = min(fl_local)     
    global_var.fracLenDict['cutposLen'].append(cutposLen)       
    #with block.suppress_stdout_stderr():
    with tf.Session() as sess:
        for kx in range(ker_size[0]):
            for ky in range(ker_size[1]):
                #import ipdb
                #ipdb.set_trace()
                tmp = tf.nn.conv2d(ifm[:,kx:int(ifm_size[0])-ker_size[0]+kx+1,ky:int(ifm_size[1])-ker_size[1]+ky+1,:],weight[kx:kx+1,ky:ky+1,:,:],strides=[1,stride,stride,1],padding='SAME')
                tmp = sess.run(tmp)
                ofm+=tmp
                ofm0 = ofm
                #ofm = fixpoint.fp(ofm,wordLen=16,fracLen=cutposLen, roundMethod='floor')._d_fp
                
    #import ipdb
    #ipdb.set_trace()
    return ofm
    
def frange(wl,value_f):
    vmin,vmax = np.min(value_f),np.max(value_f)
    il = 0
    while 2**il<vmax or -2**il>vmin:
        il += 1
    return wl-il
    
def tf_float2fx_floor(value, wl, fl):
    #with tf.Graph().as_default():
    gap = tf.constant(2.0**(-fl))
    minVal = tf.constant(-2.0**(wl-1))*gap
    maxSteps = 2.0**wl
    nSteps = tf.clip_by_value(tf.math.floordiv((value-minVal),gap),0.0,maxSteps)
    val_fp = minVal+nSteps*gap
    return val_fp
    
def tf_float2fx_round(value, wl, fl):
    #with tf.Graph().as_default():
    gap = tf.constant(2.0**(-fl))
    minVal = tf.constant(-2.0**(wl-1))*gap
    maxSteps = 2.0**wl
    nSteps = tf.clip_by_value(tf.round(tf.divide((value-minVal),gap)),0.0, maxSteps)
    val_fp = minVal+nSteps*gap
    return val_fp
    
def tf_float2fx_round_forward(value, wl, fl):
    gap = 2.0**(-fl)
    minVal = (-2.0**(wl-1))*gap
    maxSteps = 2.0**wl
    nSteps = np.clip(np.round((value-minVal)/gap),a_min=0.0, a_max=maxSteps)
    val_fp = minVal+nSteps*gap
    return np.array(val_fp)
# define a new Tensorflow Op with gradient
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Def custom square function using np.square instead of tf.square:
def tf_float2fx_round_op(value, xl, fl, name=None):
    
    with tf.name_scope(name, "tf_float2fx_round", [value, xl, fl]) as name:
        out = py_func(tf_float2fx_round_forward,
                        [value, xl, fl],
                        [tf.float32],
                        name=name,
                        grad=_LinearGrad)  # <-- here's the call to the gradient
        return out[0]

# Actual gradient:
def _LinearGrad(op, grad):
    return grad,grad,grad    
