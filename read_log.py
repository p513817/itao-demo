import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mode",default='train',help="[train, eval, prune, export, infer]")
parser.add_argument("--debug",action='store_true')
args = parser.parse_args()

if args.mode=='train':
    log_file = './log/trained.log'     
elif args.mode=='eval':
    log_file = './log/eval.log'
elif args.mode=='prune':
    log_file = './log/prune.log'
elif args.mode=='infer':
    log_file = './log/inference.log'
elif args.mode=='export':
    log_file = './log/export.log'

with open(log_file) as log:
    lines = log.readlines()
    cost_time = 1
    flag = False
    
    avg = {'epoch':None, 'avg_loss':None}
    val = {'epoch':None, 'val_loss':None}
    avg_nums, val_nums = 0, 0

    for line in lines:
        line = line.rstrip('\n')        
        if args.mode=='train':
            if 'Epoch' in line:
                _epoch = line.split(' ')[1].split('/')
                if len(_epoch)==2:
                    cur_epoch, max_epoch = _epoch
                    # if args.debug: print(f'Epoch [{cur_epoch}/{max_epoch}]')
                    avg['epoch'] = cur_epoch
                    val['epoch'] = cur_epoch
            if 'loss' in line:
                if 'Validation loss: ' in line:
                    val_loss = float(line[len('Validation loss: '): ])
                    # if args.debug: print(f'Val Loss: {val_loss}')
                    val['val_loss'] = val_loss
                else:
                    loss = float(line[line.find('- loss: ')+len('- loss: '):])
                    cost_time = float(line[line.find('- ')+len('- '):line.find('s')])
                    # if args.debug: print(f'Loss: {loss}, Cost Time: {cost_time}s')
                    avg['avg_loss'] = loss
                    time.sleep(cost_time*0.001)

            for i in avg.values():
                avg_nums = avg_nums + 1 if i==None else 0
            for i in val.values():
                val_nums = val_nums + 1 if i==None else 0
            
            if avg_nums == 0:
                if args.debug: print(avg, flush=True)
                for key in avg.keys():
                    avg[key] = None

            if val_nums == 0:
                if args.debug: print(val, flush=True)
                for key in val.keys():
                    val[key] = None

            if not args.debug: print(line, flush=True)
        elif args.mode=='eval':
            if not args.debug: print(line, flush=True)
            else:
                if '*******************************' in line:
                    flag = not flag
                else:
                    if flag:
                        print(line, flush=True)
        elif args.mode=='prune':
            if not args.debug: 
                print(line, flush=True)
                if 'INFO' in line:
                    time.sleep(1)
        elif args.mode=='export':
            if not args.debug: 
                print(line, flush=True)
                if 'INFO' in line:
                    time.sleep(1)
        elif args.mode=='infer':
            if not args.debug:
                print(line, flush=True)
                if "Start Inference" in line:
                    time.sleep(2)
        else:
            pass
                

        