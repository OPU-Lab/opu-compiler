import json
import numpy as np
import argparse
import os
import torch 
import torch.nn as nn

# Command line arguments
parser = argparse.ArgumentParser(description='generate golden results')
parser.add_argument('--config',metavar='IR', type=str, nargs='?', default='OPU_IR.json', help='path of input json')
parser.add_argument('--input',metavar='INPUT', type=str, nargs='?', default='', help='path of input fm')
parser.add_argument('--weight_dir',metavar='WDIR', type=str, nargs='?', default='dump', help='path of dumped weights')
parser.add_argument('--prelu_alpha',metavar='ALPHA', type=float, nargs='?', default='0.1', help='prelu alpha')

class GoldenGen:
    def __init__(self, args):
        self.config = args.config
        self.input = args.input
        self.wdir = args.weight_dir
        self.leaky_relu_alpha = args.prelu_alpha
        self.var_dict = {}
        self.check()
        
    def check(self):
        dat = np.load(self.input)
        self.ir = {}
        with open(self.config) as json_file: 
            lines = json_file.readlines()
            for line in lines:
                ir = json.loads(line)
                self.ir[int(ir['index'])] = ir
        self.var_dict[0] = torch.from_numpy(dat)
        pre_pad, pz = self.check_ifm_prepadding(dat, self.ir[1])
        if pre_pad:
            self.var_dict[0] = self.padding(self.var_dict[0], pz)
        
    def check_ifm_prepadding(self, x, ir):
        target_input_size = ir['input_size'] # NHWC in IR.json
        th, tw = target_input_size[1], target_input_size[2]
        n, c, h, w = x.shape
        ph = th - h
        pw = tw - w
        ph_l = ph // 2
        ph_r = ph - ph_l
        pw_l = pw // 2
        pw_r = pw - pw_l
        pad_en = ph > 0 or pw > 0
        return pad_en, (ph_l, ph_r, pw_l, pw_r)
        
    def padding(self, x, pz): # pz : [top, down, left, right]
        pad = nn.ConstantPad2d([pz[2], pz[3], pz[0], pz[1]], 0)
        print(pad)
        x = pad(x)
        return x
        
    def post_padding(self, x, ir):
        if ir['post_padding'] == 1:
            pz = ir['post_padding_size']
            x = self.padding(x, pz)
        return x
    
    def pre_remove_padding(self, x, ir):
        pz = ir['pre_remove_padding_size']
        if sum(pz) > 0:
            print('PRE REMOVE PAD',pz)
            data = x.detach().numpy()
            n,c,h,w = data.shape
            data = data[:,:,pz[0]:h-pz[1],pz[2]:w-pz[3]]
            x = torch.from_numpy(data)
        return x
        
    def pooling(self, x, ir):
        kernel_size = ir['pooling_size'][1:3]
        stride = ir['pooling_stride'][1:3]
        if ir['pooling_type'] == 1:
            maxpool2d = nn.MaxPool2d(kernel_size, stride)
            print(maxpool2d)
            x = maxpool2d(x)
        elif ir['pooling_type'] == 2:
            #avgpool2d = nn.AvgPool2d(kernel_size, stride)
            avgpool2d = nn.AvgPool2d(x.detach().numpy().shape[-2:])
            print(avgpool2d)
            x = avgpool2d(x)
        return x
            
    def activation(self, x, ir):
        act_opcode = ir['activation_type']
        if act_opcode == 1:
            relu = nn.ReLU()
            print(relu)
            x = relu(x)
        elif act_opcode == 2:
            leaky_relu = nn.LeakyReLU(negative_slope=self.leaky_relu_alpha, inplace=True)
            print(leaky_relu)
            x = leaky_relu(x)
        elif act_opcode > 0:
            assert 0, act_opcode
        return x
    
    def upsampling(self, x, ir):
        if not ir['upsample'] == 0:
            scale = ir['upsample']
            method = ir['upsample_method']
            if method == 'nearest_neighbor':
                mode = 'nearest'
            else:
                raise NotImplementedError
            upsample = nn.Upsample(scale_factor=scale, mode=mode)
            print(upsample)
            x = upsample(x)
        return x
        
    def residual_add(self, x, ir):
        if ir['residual'] == 1:
            print('ResidualAdd with output of LAYER ', ir['residual_source'][0]-1)
            for ii in ir['residual_source']:
                # check shape
                residue = self.var_dict[ii]
                if not residue.shape == x.shape:
                    pu, pd, pl, pr = self.ir[ii - 1]['post_padding_size']
                    print('Residuel src layer', ii - 1, 'remove padding', (pu, pd, pl, pr))
                    residue = residue[:,:,pu:-pd,pl:-pr]
                x += residue
        return x        
    
    def post_order(self, x, ir):
        order = ir['res_position']
        if order == 3:
            x = self.activation(x, ir)
            x = self.pooling(x, ir)
        elif order == 2:
            x = self.activation(x, ir)
            x = self.pooling(x, ir)
            x = self.residual_add(x, ir)            
        elif order == 1:
            x = self.activation(x, ir)
            x = self.residual_add(x, ir)
            x = self.pooling(x, ir)
        elif order == 0:
            x = self.residual_add(x, ir)
            x = self.activation(x, ir)
            x = self.pooling(x, ir)
        else:
            raise NotImplementedError
        return x
        
    def check_tensor_shape(self, ir_shape, pt_tensor):
        # ir_shape in NHWC (self-defined), pt_shape in NCHW (PyTorch)
        pt_shape = list(pt_tensor.detach().numpy().shape)
        if pt_tensor.ndim < 4:
            for ii in range(4 - pt_tensor.ndim):
                pt_shape.append(1)
        assert len(pt_shape) == len(ir_shape), "unmatched ndim"
        def to_str_list(x):
            return [str(u) for u in x]
        err_msg = ','.join(to_str_list(ir_shape)) + ' v.s. ' + ','.join(to_str_list(pt_shape))
        assert ir_shape[0] == pt_shape[0], err_msg
        assert ir_shape[1] == pt_shape[2], err_msg
        assert ir_shape[2] == pt_shape[3], err_msg
        assert ir_shape[3] == pt_shape[1], err_msg
    
    def quantize(self, x, ir, info):
        self.quantize_fm = True
        self.fm_dw = 8
        if self.quantize_fm:
            if info == 'input':
                fl = ir['input_fraclen']
            elif info == 'output':
                fl = ir['output_fraclen']
            else:
                assert 0
            base = 2 ** (-fl)
            dat = x.detach().numpy()
            dat = np.maximum(-2**self.fm_dw, np.minimum(2**self.fm_dw - 1, np.floor(dat / base))) * base
            x = torch.from_numpy(dat)
            print('quantize', info, ' fraclen:', fl)
        return x
        
    def run_layer(self, ir, params):        
        if len(params['input']) == 1:
            x = params['input'][0]
        else:
            x = torch.cat(params['input'], 1)
        print('input_shape:', x.shape)
        self.check_tensor_shape(ir['input_size'], x)
        x = self.pre_remove_padding(x, ir)
        x = self.quantize(x, ir, 'input')
        opcode = ir['type']
        if opcode == 1:
            in_channels = ir['input_size'][-1]
            out_channels  = ir['output_size'][-1]
            kernel_size = tuple(ir['ker_size'][1:3])
            stride = tuple(ir['ker_stride'][1:3])
            groups = ir['group']
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          groups=groups, bias=False)
            print(conv2d)
            # hwio -> oihw
            weight = torch.from_numpy(params['weight'].transpose(3,2,0,1))
            conv2d.weight = nn.Parameter(weight)
            x = conv2d(x)
            bias = params['bias']
            bias = np.expand_dims(bias, bias.ndim)
            bias = np.expand_dims(bias, bias.ndim)
            bias = torch.from_numpy(bias)
            x = x + bias
            #import ipdb;ipdb.set_trace()
        elif opcode == 0:
            ci = np.prod(ir['input_size'])
            co = ir['output_size'][-1]
            x = x.view(-1, ci)
            weight = torch.from_numpy(params['weight'])
            fc = nn.Linear(ci, co)
            fc.weight = nn.Parameter(weight)
            bias = torch.from_numpy(params['bias'])
            fc.bias = nn.Parameter(bias)
            print(fc)
            x = fc(x)
        else:
            assert 0
        x = self.post_order(x, ir)
        x = self.upsampling(x, ir)
        x = self.post_padding(x, ir)
        print('output_shape:', x.shape)
        self.check_tensor_shape(ir['output_size'], x)
        x = self.quantize(x, ir, 'output')
        self.var_dict[int(ir['index'])] = x
        np.save('golden_layer_'+str(int(ir['index']))+'_out.npy', x.detach().numpy())
        print()
    
    def save_output(self):
        output = []
        for i in range(len(self.ir)):
            if -1 in self.ir[i+1]['output_layer']:
                dat = self.var_dict[i+1].detach().numpy()
                output.append(dat)
        output.reverse()
        if len(output) > 1:
            tmp = [dat.reshape(dat.shape[0], dat.shape[1], -1) for dat in output]
            tmp.sort(key=lambda x:x.size)
            out = np.concatenate(tmp, axis=2) # specific for yolov3
        else:
            out = output[0]
        np.save('golden_output_ir.npy', out)
        
    def run(self):
        layer_num = len(self.ir)
        for i in range(layer_num):
            print('LAYER', i)
            params = {}
            print('inputs', self.ir[i+1]['input_layer'])
            # collect inputs
            params['input'] = [self.var_dict[x] for x in self.ir[i+1]['input_layer']]
            path_w = os.path.join(self.wdir, 'weights_'+str(i)+'.npy')
            params['weight'] = np.load(path_w)
            path_b = os.path.join(self.wdir, 'bias_'+str(i)+'.npy')
            params['bias'] = np.load(path_b)
            # run
            self.run_layer(self.ir[i + 1], params)
        self.save_output()
        
args = parser.parse_args()           
GoldenGen(args).run()