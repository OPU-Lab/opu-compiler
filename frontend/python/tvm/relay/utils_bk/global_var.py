import numpy as np
from collections import defaultdict
import os

class global_var:
    def __init__(self):
        self.fracLenDict = defaultdict(list)
        '''self.fx_output_dir = '/home/tiandong/tvm/example/tests/vgg19/output/'
        if not os.path.exists(self.fx_output_dir): os.mkdir(self.fx_output_dir)
        if os.path.exists(self.fx_output_dir+'fracLen.txt'): os.remove(self.fx_output_dir+'fracLen.txt')'''
    
    def dump(self, path):
        with open(path,'w') as f:
            print('fix_bias_fi_pos = [', file=f,end='')
            for i in range(len(self.fracLenDict['bias'])):
                print(self.fracLenDict['bias'][i],end='',file=f)
                if i==len(self.fracLenDict['bias'])-1:
                    print('];',file=f)
                else:
                    print(', ',end='',file=f)
            print('fix_weights_fi_pos = [', file=f,end='')
            for i in range(len(self.fracLenDict['weight'])):
                print(self.fracLenDict['weight'][i],end='',file=f)
                if i==len(self.fracLenDict['weight'])-1:
                    print('];',file=f)
                else:
                    print(', ',end='',file=f)
            print('fix_fm_pos_ori = [', file=f,end='')
            for i in range(len(self.fracLenDict['fm'])):
                print(self.fracLenDict['fm'][i],end='',file=f)
                if i==len(self.fracLenDict['fm'])-1:
                    print('];',file=f)
                else:
                    print(', ',end='',file=f)
        path = '/'.join(path.split('/')[:-1])+'/cutposLen.txt'
        with open(path,'w') as f:
            print('cutposLen = [', file=f,end='')
            for i in range(len(self.fracLenDict['cutposLen'])):
                print(self.fracLenDict['cutposLen'][i],end='',file=f)
                if i==len(self.fracLenDict['cutposLen'])-1:
                    print('];',file=f)
                else:
                    print(', ',end='',file=f)