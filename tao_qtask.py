from PyQt5.QtCore import QThread, pyqtSignal
import subprocess

class TAO_Train(QThread):

    trigger = pyqtSignal(object)

    def __init__(self):
        super(TAO_Train, self).__init__()
        self.flag = True        
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}
        self.nums = 0

    def run(self):
        proc = subprocess.Popen(["python3", "read_log.py", "train"], stdout=subprocess.PIPE)    
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n')
                self.nums = 0
                if 'Epoch ' in line:
                    line_cnt = line.split(' ')
                    if len(line_cnt)==2:  
                        self.data['epoch'] = int(line_cnt[1].split('/')[0])
                    continue
                elif 'loss: ' in line:
                    if 'Validation' in line:           
                        self.data['val_loss']= round(float( line.split('loss: ')[1]), 3)                   
                    else:
                        self.data['avg_loss']= round(float( line.split('loss: ')[1]), 3)
                else:
                    continue

                for i in self.data.values():
                    if i == None:continue
                    self.nums = self.nums + 1 

                if self.nums>=2:
                    self.trigger.emit(self.data)
                    if self.nums>=3:
                        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}
        self.trigger.emit({})    
            
    def stop(self):
        self.flag=False

class TAO_VAL(QThread):

    trigger = pyqtSignal(str)

    def __init__(self):
        super(TAO_VAL, self).__init__()
        self.flag = True  
        self.trg = False      
        self.sym = '*******************************'

    def run(self):
        proc = subprocess.Popen(["python3", "read_log.py", "eval"], stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n')
                if self.sym in line: 
                    self.trg = not self.trg
                else:
                    if self.trg: self.trigger.emit(line)
        self.trigger.emit("end")

    def stop(self):
        self.flag=False

class TAO_PRUNE(QThread):

    trigger = pyqtSignal(str)

    def __init__(self):
        super(TAO_PRUNE, self).__init__()
        self.flag = True  

    def run(self):
        proc = subprocess.Popen(["python3", "read_log.py", "prune"], stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            else:
                for line in proc.stdout:
                    line = line.decode("utf-8").rstrip('\n')
                    if "[INFO]" in line:
                        self.trigger.emit(line)
        self.trigger.emit("end")
        
    def stop(self):
        self.flag=False

class TAO_INFER(QThread):

    trigger = pyqtSignal(dict)
    info = pyqtSignal(str)

    def __init__(self, epoch=50):
        super(TAO_INFER, self).__init__()
        self.epoch = epoch
        self.flag = True        
        self.data = {}
        self.trg = False
        self.cur_name = ""
    def run(self):
        proc = subprocess.Popen(["python3", "read_log.py", "infer"], stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n').strip()
                # print(line)
                if "[INFO]" in line:
                    self.info.emit(line)

                if ":{" in line:
                    self.cur_name = line.replace('"','').replace(':','').replace('{','')
                    self.data[self.cur_name] = []
                    self.trg = True
                elif "}" in line:
                    self.trg = False
                    self.trigger.emit(self.data)
                else:
                    if self.trg:
                        self.data[self.cur_name].append(line.rstrip(" ").rstrip(","))
        self.trigger.emit({})

    def stop(self):
        self.flag=False

class TAO_RETRAIN(QThread):

    trigger = pyqtSignal(dict)

    def __init__(self, epoch=50):
        super(TAO_RETRAIN, self).__init__()
        self.epoch = epoch
        self.flag = True 
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}
        self.nums = 0     

    def run(self):
        proc = subprocess.Popen(["python3", "gen_train_log.py", "-e", f"{self.epoch}"], stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n')
                _epoch, _avg, _val = line.split(',')
                self.data['epoch'] = int(_epoch.split(': ')[1])
                self.data['avg_loss'] = float(_avg.split(': ')[1])
                self.data['val_loss'] = float(_val.split(': ')[1])
                
                self.trigger.emit(self.data)
        self.trigger.emit({})

    def stop(self):
        self.flag=False

class TAO_EXPORT(QThread):

    trigger = pyqtSignal(str)

    def __init__(self):
        super(TAO_EXPORT, self).__init__()
        self.flag = True  

    def run(self):
        proc = subprocess.Popen(["python3", "read_log.py", "export"], stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n')
                if "[INFO]" in line:
                    self.trigger.emit(line)
        self.trigger.emit("end")

    def stop(self):
        self.flag=False



