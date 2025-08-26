import sys
import os
import copy
import json

opt = dict()

def generate_command(opt):
    cmd = 'python3 main.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

# main experiment
for data in ['cora','citeseer','C-M10-M']:
    for opt['seed'] in range(1):
        opt['dataset'] = data
        if data == 'cora':
            opt['train-class'] = 3
            opt['val-class'] = 0
            run(opt)
            opt['train-class'] = 2
            opt['val-class'] = 2
            run(opt)
        elif data == 'citeseer':
            opt['train-class'] = 2
            opt['val-class'] = 0
            run(opt)
            opt['train-class'] = 2
            opt['val-class'] = 2
            run(opt)
        elif data == 'C-M10-M':
            opt['train-class'] = 3
            opt['val-class'] = 0
            run(opt)
            opt['train-class'] = 2
            opt['val-class'] = 2
            run(opt)
        else:
            print("Please add the dataset!")


