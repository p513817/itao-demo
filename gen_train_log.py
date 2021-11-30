import multiprocessing as mp
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch",type=int, default=5)
args = parser.parse_args()

FLAG=True
EPOCH=args.epoch
CUR_EPOCH=0
AVG_LOSS=10
VAL_LOSS=1

FORCE_AVG=1
FORCE_VAL=0.01

def worker(iter=0.001):
    
    global AVG_LOSS, VAL_LOSS, FORCE_AVG, FORCE_VAL, CUR_EPOCH
    
    while(CUR_EPOCH<EPOCH):

        rand = np.random.randn(1)
        rand_temp = np.random.rand(1)
        rand_time = np.random.randint(1,100)

        rand = rand*0.4 if rand > 0 else rand*0.1
        rand_temp = rand_temp*0.2 if rand_temp > 0 else rand_temp*0.02

        AVG_LOSS = float(AVG_LOSS-(rand*FORCE_AVG))
        VAL_LOSS = float(VAL_LOSS-(rand_temp*FORCE_VAL))
        print('Epoch: {}, Avg Loss: {}, Val Loss: {} '.format(CUR_EPOCH+1, AVG_LOSS, VAL_LOSS))

        if AVG_LOSS<=(AVG_LOSS/10):
            FORCE_AVG=FORCE_AVG/10
        elif AVG_LOSS<=(AVG_LOSS/100):
            FORCE_AVG=FORCE_AVG/100
        elif AVG_LOSS<=(0.1):
            FORCE_AVG=0.00005
        elif AVG_LOSS<=(0.01):
            FORCE_AVG=0.000000001
        if VAL_LOSS<=(VAL_LOSS/10):
            FORCE_VAL=FORCE_VAL/10
        elif VAL_LOSS<=(VAL_LOSS/100):
            FORCE_VAL=FORCE_VAL/100
        elif VAL_LOSS<=(0.01):
            FORCE_VAL=0.00001
        elif VAL_LOSS<=(0.0001):
            FORCE_VAL=0.000000001

        CUR_EPOCH = CUR_EPOCH +1 
        time.sleep(iter*rand_time)
if __name__ == '__main__':

    proc = mp.Process(target=worker, args=())
    proc.start()
    proc.join()
    