#!/usr/bin/python3
# -*- coding: utf-8 -*-
import PyQt5
from PyQt5 import QtWidgets, QtGui, uic,QtCore
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
import sys
import time
import os
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from typing import *
import pyqtgraph as pg

""" 自己的函式庫自己撈"""
from tao_configs import TRAIN_CONF, RETRAIN_CONF, PRUNE_CONF, OPT, ARCH_LAYER
from tao_qtask import TAO_PRUNE, TAO_RETRAIN, TAO_Train, TAO_VAL, TAO_INFER, TAO_EXPORT
from PyQt5.QtCore import QTimer, QPropertyAnimation

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
MODEL_ROOT = './model'
PRUNED_ROOT = './pruned'
INFER_IMG_ROOT = './infer_images'
INFER_LBL_ROOT = './infer_labels'

class UI(QtWidgets.QMainWindow):

    ####################################################################################################
    #                                               INIT                                               #
    ####################################################################################################

    def __init__(self):
        super(UI, self).__init__() # Call the inherited classes __init__ method
        self.ui = uic.loadUi("itao_v0.2.ui", self) # Load the .ui file
        self.ui.setWindowTitle('iTAO')
        self.first_page_id = 0  # 1-1 = 0
        self.end_page_id = 3    # 4-1 = 0
        
        """ 將元件統一 """
        font = QtGui.QFont("Consolas", 10)
        self.ui.setFont(font)
        self.space = len('learning_rate') # get longest width in console
        self.page_buttons_status={0:[0,0], 1:[1,0], 2:[1,0], 3:[1,0]}
        self.tabs = [ self.tab_1, self.tab_2, self.tab_3, self.tab_4 ]
        self.progress = [ self.ui.t1_progress, self.ui.t2_progress, self.ui.t3_progress, self.ui.t4_progress]
        self.frames = [None, self.ui.t2_frame, self.ui.t3_frame, None]
        self.consoles = [ self.ui.t1_console, self.ui.t2_console, self.ui.t3_console, self.ui.t4_console]
        [ self.tabs[i].setEnabled(True) for i in range(len(self.tabs))] # This is for debug

        """ 建立 & 初始化 Tab2 跟 Tab3 的圖表 """
        self.pws = [None, pg.PlotWidget(self), pg.PlotWidget(self), None]
        self.pw_lyrs = [None, QVBoxLayout(), QVBoxLayout(), None]
        [ self.init_plot(i) for i in range(1,3) ]

        """ 設定 Previous、Next 的按鈕 """
        self.current_page_id = self.first_page_id
        self.ui.main_tab.setCurrentIndex(self.first_page_id)
        self.ui.bt_next.clicked.connect(self.ctrl_page_event)
        self.ui.bt_previous.clicked.connect(self.ctrl_page_event)
        self.info = ""
        self.update_page_button()
        self.ui.main_tab.currentChanged.connect(self.update_page_button)

        ####################################################################################################
        #                                               TAB                                                #
        ####################################################################################################

        """ ######## 設定 Tab 1 的相關資訊 ######## """
        self.t1_objects = [ self.ui.t1_combo_task, self.ui.t1_combo_model , self.ui.t1_combo_bone , self.ui.t1_combo_layer , self.ui.t1_bt_dset ,self.ui.t1_bt_label ]
        self.sel_idx = [-1,-1,-1,-1,-1,-1]

        self.ui.t1_combo_task.currentIndexChanged.connect(self.get_task)
        self.ui.t1_combo_model.currentIndexChanged.connect(self.get_model)
        self.ui.t1_combo_bone.currentIndexChanged.connect(self.get_backbone)
        self.ui.t1_combo_layer.currentIndexChanged.connect(self.get_nlayer)
        self.ui.t1_bt_dset.clicked.connect(self.get_folder)
        self.ui.t1_bt_label.clicked.connect(self.get_file)

        """ ######## 設定 Tab 2 的相關資訊 ######## """
        self.t2_var = { "avg_epoch":[],
                        "avg_loss":[],
                        "val_epoch":[],
                        "val_loss":[] }
        self.worker, self.worker_eval = None, None
        self.ui.t2_bt_train.clicked.connect(self.t2_train_event)
        self.ui.t2_bt_stop.clicked.connect(self.t2_stop_event)

        """ ######## 設定 Tab 3 的相關資訊 ######## """
        self.worker_prune, self.worker_retrain = None, None
        self.t3_bt_pruned.clicked.connect(self.t3_prune_event)
        self.t3_bt_retrain.clicked.connect(self.t3_retrain_event)
        self.t3_bt_stop.clicked.connect(self.t3_stop_event)

        self.t3_var = { "avg_epoch":[],
                        "avg_loss":[],
                        "val_epoch":[],
                        "val_loss":[] }

        self.prune_log_key = [  "['nvcr.io']",
                                "Exploring graph for retainable indices",
                                "Pruning model and appending pruned nodes to new graph",
                                "Pruning ratio (pruned model / original model)",
                                "Stopping container" ]

        """ ######## 設定 Tab 4 的相關資訊 ######## """
        self.precision_radio = {"INT8":self.t4_int8, "FP16":self.t4_fp16, "FP32":self.t4_fp32}
        
        self.worker_infer, self.export_name, self.precision = None, None, None
        self.infer_files, self.saving_folder = None, None

        self.ui.t4_bt_upload.clicked.connect(self.get_file)
        self.ui.t4_bt_savepath.clicked.connect(self.get_folder)

        self.t4_bt_infer.clicked.connect(self.t4_infer_event)
        self.t4_bt_export.clicked.connect(self.export_event)

        self.export_log_key = [  "Registry: ['nvcr.io']",
                                "keras_exporter",
                                "keras2onnx",
                                "Stopping container" ]

        self.ui.t4_bt_pre_infer.clicked.connect(self.ctrl_result_event)
        self.ui.t4_bt_next_infer.clicked.connect(self.ctrl_result_event)
        self.ls_infer_name, self.ls_infer_label = [], []
        self.cur_pixmap = 0

    ####################################################################################################
    #                                           SUB FUNCTION                                           #
    ####################################################################################################

    """ 取得資料夾路徑 """
    def get_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
        if self.current_page_id==0:
            TRAIN_CONF['dataset_path'] = folder_path
            self.sel_idx[4]=1 
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))
        elif self.current_page_id==3:
            self.saving_folder = folder_path
            
    """ 取得檔案路徑 """
    def get_file(self):
        filename, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./")
        if self.current_page_id==0:
            TRAIN_CONF['label_path'] = filename
            self.sel_idx[5]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))
        elif self.current_page_id==3:
            self.infer_files = filename
              
    """ 更新頁面與按鈕 """
    def update_page_button(self):
        idx = self.ui.main_tab.currentIndex()
        self.current_page_id = idx
        self.bt_previous.setEnabled(self.page_buttons_status[self.current_page_id][0])
        self.bt_next.setEnabled(self.page_buttons_status[self.current_page_id][1])
    
    """ 更新頁面的事件 next, previous 按鈕 """
    def ctrl_page_event(self):
        trg = self.sender().text().lower()
        if trg=="next":
            if self.current_page_id < self.end_page_id :
                self.current_page_id = self.current_page_id + 1
                self.ui.main_tab.setCurrentIndex(self.current_page_id)
        else:   # previous
            if self.current_page_id > self.first_page_id :
                self.current_page_id = self.current_page_id - 1
                self.ui.main_tab.setCurrentIndex(self.current_page_id)
        
    """ 初始化圖表 """
    def init_plot(self, idx=None, xlabel="Epochs", ylabel="Loss"):
        if idx==None:   idx = self.current_page_id
        self.pws[idx].showGrid(x=True, y=True)
        self.pws[idx].addLegend(offset=(0., .5))
        self.pws[idx].setLabel("left", ylabel)
        self.pws[idx].setLabel("bottom", xlabel)
        self.frames[idx].setLayout(self.pw_lyrs[idx])
        self.pw_lyrs[idx].addWidget(self.pws[idx])
        self.pws[idx].hide()                            # 先關閉等待 init_console 的時候才開

    """ 初始化 Console """
    def init_console(self):
        self.consoles[self.current_page_id].clear()    
        self.pws[self.current_page_id].show()
        if self.current_page_id in [1,2]:
            self.pws[self.current_page_id].clear()
            if self.current_page_id==1: [val.clear() for _, val in self.t2_var.items() ]
            if self.current_page_id==2: [val.clear() for _, val in self.t3_var.items() ]

    """ 更新進度條，如果進度條滿了也會有對應對動作 """
    def update_progress(self, idx, cur, limit):
        val = int(cur*(100/limit))
        self.progress[idx].setValue(val)
        if val>=100:
            self.page_finished_event()
        
    """ 進度條滿了 -> 頁面任務完成 -> 對應對動作 """
    def page_finished_event(self):
        if self.current_page_id==0:
            self.insert_text("SHOW TRAIN CONFIG", TRAIN_CONF)
            self.swith_page_button(previous=0, next=1)
        # elif self.current_page_id==1:
        #     pass
        # elif self.current_page_id==2:
        #     pass
        # elif self.current_page_id==3:
        #     pass

    """ 於對應的區域顯示對應的配置檔內容 """
    def insert_text(self, title, config=None):
        console = self.consoles[self.current_page_id]
        log = [ 
            "-"*((console.width()-6)//4), "\n", 
            f"{title}\n", "\n" 
        ]
        [ console.insertPlainText(content) for content in log ]
        if config is not None:            
            [ console.insertPlainText(f"{key:<{self.space}}: {val}\n") for key, val in config.items()]

    """ 用於修改各自頁面的狀態 """
    def swith_page_button(self, previous, next=None):
        self.page_buttons_status[self.current_page_id][:] = [previous, next if next !=None else previous]
        self.bt_previous.setEnabled(self.page_buttons_status[self.current_page_id][0])
        self.bt_next.setEnabled(self.page_buttons_status[self.current_page_id][1] if next != None else self.page_buttons_status[self.current_page_id][0])

    """ 移動到最後一行"""
    def mv_last_line(self):
        self.consoles[self.current_page_id].textCursor().movePosition(QtGui.QTextCursor.Start)  # 將位置移到LOG最下方 (1)
        self.consoles[self.current_page_id].ensureCursorVisible()                               # 將位置移到LOG最下方 (2)

    ####################################################################################################
    #                                                T4                                                #
    ####################################################################################################
    
    """ T4 -> 匯出的事件 """
    def export_event(self):
        self.worker_export = TAO_EXPORT()
        self.worker_export.start()
        self.worker_export.trigger.connect(self.update_export_log)
        self.init_console()
        self.insert_text("Start Export ... ")
        self.export_name = self.t4_etlt_name.toPlainText()
        self.precision = self.check_radio()
        self.consoles[self.current_page_id].insertPlainText(f"Export Path : {self.export_name}\n") 
        self.consoles[self.current_page_id].insertPlainText(f"Precision : {self.precision}\n") 
    
    """ T4 -> 檢查 radio 按了哪個 """
    def check_radio(self):
        for precision, radio in self.precision_radio.items():
            if radio.isChecked(): return precision
        return ''

    """ T4 -> 按下 Inference 按鈕之事件 """
    def t4_infer_event(self):
        self.worker_infer = TAO_INFER()
        self.worker_infer.start()
        self.worker_infer.trigger.connect(self.update_infer_log)
        self.worker_infer.info.connect(self.update_infer_log)
        self.init_console()
        self.insert_text("Start Inference ... ")
        self.consoles[self.current_page_id].insertPlainText(f"Saving Path: {self.saving_folder}\n")
        self.consoles[self.current_page_id].insertPlainText(f"Input Data: {self.infer_files}\n")

    """ T4 -> 更新 Inference 的資訊 """
    def update_infer_log(self, data):
        if bool(data):
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
        else:
            self.ui.t4_bt_next_infer.setEnabled(True)
            self.ui.t4_bt_pre_infer.setEnabled(True)
            self.insert_text("Inference ... finished !!! ")    
            self.load_result()
            self.worker_infer.quit()
            self.swith_page_button(True)

    """ T4 -> 控制顯示結果的事件 """
    def ctrl_result_event(self):
        who = self.sender().text()
        if who=="<":
            if self.cur_pixmap > 0: self.cur_pixmap = self.cur_pixmap - 1
            self.show_result()
        elif who==">":
            if self.cur_pixmap < len(self.ls_infer_name)-1: self.cur_pixmap = self.cur_pixmap + 1
            self.show_result()
        else:
            pass
    
    """ T4 -> 將 pixmap、title、log 顯示出來， """
    def show_result(self):
        pixmap = QtGui.QPixmap(self.ls_infer_name[self.cur_pixmap])
        self.t4_frame.setPixmap(pixmap.scaled(self.frame_size-10, self.frame_size-10))
        self.t4_infer_name.setText(self.ls_infer_name[self.cur_pixmap])
        self.insert_text(self.ls_infer_name[self.cur_pixmap])
        [ self.consoles[self.current_page_id].insertPlainText(f"[{idx}] {cnt}") for idx,cnt in enumerate(self.ls_infer_label[self.cur_pixmap]) ]
        self.mv_last_line()

    """ T4 -> 將名字與標籤檔儲存下來，方便後續調用 """
    def load_result(self):
        # 更新大小，在這裡更新才會是正確的大小
        self.frame_size = self.t4_frame.width() if self.t4_frame.width()<self.t4_frame.height() else self.t4_frame.height()
        # 更新路徑        
        trg_folder = INFER_IMG_ROOT if self.saving_folder==None else self.saving_folder
        # 把所有的檔案給 Load 進 ls_infer_name
        self.cur_pixmap = 0
        for file in self.infer_files:
            base_name = os.path.basename(file)
            # 儲存名稱的相對路徑
            self.ls_infer_name.append(os.path.join( trg_folder, base_name ))
            # 儲存標籤檔的相對路徑
            label_name = os.path.splitext(os.path.join( trg_folder, base_name ))[0]+'.txt'
            with open(label_name, 'r') as lbl:
                content = lbl.readlines()
                self.ls_infer_label.append(content)
        # show first one
        self.cur_pixmap = 0
        self.show_result()
    
    """ T4 -> 更新輸出的LOG """
    def update_export_log(self, data):
        if data != "end":
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
            self.mv_last_line()
            for key in self.export_log_key:
                if key in data: self.update_progress(self.current_page_id, self.export_log_key.index(key)+1, len(self.export_log_key))  
        else:
            self.insert_text("Export ... finished !!! ")
            self.worker_export.quit()
            self.swith_page_button(True)

    ####################################################################################################
    #                                                T3                                                #
    ####################################################################################################

    """ T3 -> 按下停止的事件 """
    def t3_stop_event(self):
        # 如果 worker_prune 有在進行的話，就強置中斷
        if self.worker_prune != None:
            self.worker_prune.terminate()
            self.ui.t3_bt_pruned.setEnabled(True)
        # 如果 worker_retrain 有在進行的話，就強置中斷
        if self.worker_retrain != None:
            self.worker_retrain.terminate()
            self.ui.t3_bt_retrain.setEnabled(True)
        # 都停止後將 page_button 開啟
        self.swith_page_button(True)
        self.ui.t3_bt_stop.setEnabled(False)

    """ T3 -> Prune 的事件 """
    def t3_prune_event(self):
        self.swith_page_button(False)
        self.ui.t3_bt_pruned.setEnabled(False)
        self.ui.t3_bt_stop.setEnabled(True)

        self.init_console()
        self.update_prune_conf()
        self.insert_text("Start Pruning Model ... ")
        self.worker_prune = TAO_PRUNE()
        self.worker_prune.start()
        self.worker_prune.trigger.connect(self.update_prune_log)

    """ T3 -> 更新 Prune 的資訊 """
    def update_prune_log(self, data):
        if data != "end":
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
            self.mv_last_line()
            for key in self.prune_log_key:
                if key in data:
                    self.update_progress(self.current_page_id, self.prune_log_key.index(key)+1, len(self.prune_log_key))            
        else:
            self.ui.t3_bt_pruned.setEnabled(True)  
            self.ui.t3_bt_retrain.setEnabled(True)
            self.ui.t3_bt_stop.setEnabled(False)
            self.swith_page_button(1,0)
            self.worker_prune.quit()
            self.insert_text("Pruning Model ... Finished !!!")
            self.pruned_compare()
    
    """ T3 -> 剪枝後計算模型大小以及顯示直方圖 """
    def pruned_compare(self):

        org_size = float(os.path.getsize(PRUNE_CONF['input_model']))/1024/1024 if PRUNE_CONF['input_model'] != 'debug mode' else 243
        aft_size = float(os.path.getsize(PRUNE_CONF['output_name']))/1024/1024 if PRUNE_CONF['output_name'] != 'debug mode' else 91.2

        self.consoles[self.current_page_id].insertPlainText(f"Unpruned Model Size : {(org_size):.3f} MB\n")
        self.consoles[self.current_page_id].insertPlainText(f"Pruned Model Size : {(aft_size):.3f} MB\n")
        self.insert_text(f"Pruning Rate : {(aft_size/org_size)*100:.3f}%\n")
        self.mv_last_line()

        self.pws[self.current_page_id].addItem(pg.BarGraphItem(x=[1], height=[org_size], width=0.5, brush='b', name="Unpruned"))
        self.pws[self.current_page_id].addItem(pg.BarGraphItem(x=[2], height=[aft_size], width=0.5, brush='g', name="Pruned"))

    """ T3 -> Retrain Event"""
    def t3_retrain_event(self):
        self.swith_page_button(False)
        self.ui.t3_bt_pruned.setEnabled(False)
        self.ui.t3_bt_retrain.setEnabled(False)
        self.ui.t3_bt_stop.setEnabled(True)
        self.update_retrain_conf()
        self.init_console()
        self.init_plot()
        self.pws[self.current_page_id].setXRange(0, int(RETRAIN_CONF['epoch']), 0.05)
        self.insert_text("Start Re-Train Model ... ")
        self.worker_retrain = TAO_RETRAIN(int(RETRAIN_CONF['epoch']))
        self.worker_retrain.start()
        self.worker_retrain.trigger.connect(self.update_retrain_log)

    """ T3 -> 更新 Retrain 的相關資訊 """
    def update_retrain_log(self, data):
        if bool(data):
            cur_epoch, avg_loss, val_loss, max_epoch = data['epoch'], data['avg_loss'], data['val_loss'], int(RETRAIN_CONF['epoch'])
            log = "{} {} {}\n".format(  f'[{cur_epoch:03}/{max_epoch:03}]',
                                        f'AVG_LOSS: {avg_loss:06.3f}',
                                        f'VAL_LOSS: {val_loss:06.3f}' if val_loss is not None else ' ')

            self.t3_var["val_epoch"].append(cur_epoch)
            self.t3_var["val_loss"].append(val_loss)        
            self.t3_var["avg_epoch"].append(cur_epoch)
            self.t3_var["avg_loss"].append(avg_loss)

            self.pws[self.current_page_id].clear()                                                  # 清除 Plot
            self.consoles[self.current_page_id].insertPlainText(log)                                # 插入內容
            self.mv_last_line()

            self.pws[self.current_page_id].plot(self.t3_var["avg_epoch"], self.t3_var["avg_loss"], pen=pg.mkPen('r', width=2), name="average loss")
            self.pws[self.current_page_id].plot(self.t3_var["val_epoch"], self.t3_var["val_loss"], pen=pg.mkPen('b', width=2), name="validation loss")
            self.update_progress(self.current_page_id, cur_epoch, max_epoch)            

        else:
            self.insert_text("Re-Train Model ... Finished !!!")
            self.worker_retrain.quit()
            self.ui.t3_bt_retrain.setEnabled(True)  
            self.ui.t3_bt_pruned.setEnabled(True)
            self.ui.t3_bt_stop.setEnabled(False)
            self.swith_page_button(1)
            

    """ T3 -> 將QT中的PRUNE配置內容映射到PRUNE_CONF """
    def update_prune_conf(self):
        RETRAIN_CONF['key'] = PRUNE_CONF['key'] = TRAIN_CONF['key']
        self.ui.t3_pruned_key.setText(TRAIN_CONF['key'])
        
        if self.ui.t3_pruned_in_model.toPlainText() == '':
            PRUNE_CONF['input_model'] = 'debug mode'
        else:
            if "{epoch}" in self.ui.t3_pruned_in_model.toPlainText():
                last_model = sorted( [ i for i in os.listdir(MODEL_ROOT)] )[-1]
                PRUNE_CONF['input_model'] = os.path.join(MODEL_ROOT, last_model)  
            else:
                PRUNE_CONF['input_model'] = os.path.join(MODEL_ROOT, self.ui.t3_pruned_in_model.toPlainText())            

        if self.ui.t3_pruned_out_name.toPlainText() == '':
            PRUNE_CONF['output_name'] = 'debug mode'
        else:
            if "{epoch}" in self.ui.t3_pruned_out_name.toPlainText():
                last_model = sorted( [ i for i in os.listdir(PRUNED_ROOT)] )[-1]
                PRUNE_CONF['output_name'] = os.path.join(PRUNED_ROOT, last_model)         
            else:
                PRUNE_CONF['output_name'] = os.path.join(PRUNED_ROOT, self.ui.t3_pruned_out_name.toPlainText())
            
        PRUNE_CONF['thres'] = self.ui.t3_pruned_threshold.value()
        self.insert_text("SHOW PRUNED CONFIG", PRUNE_CONF)

    """ T3 -> 將QT中的RETRAIN配置內容映射到RETRAIN_CONF """
    def update_retrain_conf(self):

        self.ui.t3_retrain_key.setText(TRAIN_CONF['key'])
        RETRAIN_CONF['output_name'] = self.ui.t3_retrain_out_model.toPlainText()
        RETRAIN_CONF['epoch'] = self.ui.t3_retrain_epoch.toPlainText()
        RETRAIN_CONF['batch_size'] = self.ui.t3_retrain_bsize.toPlainText()
        RETRAIN_CONF['learning_rate'] = self.ui.t3_retrain_lr.toPlainText()
        RETRAIN_CONF['custom'] = self.ui.t3_retrain_c1.toPlainText()
        self.insert_text("SHOW RETRAINED CONFIG", RETRAIN_CONF)

    ####################################################################################################
    #                                                T2                                                #
    ####################################################################################################

    """ T2 -> 當按下 train 按鈕的時候進行的事件 """
    def t2_train_event(self):
        self.update_train_conf()
        self.init_console()
        self.pws[self.current_page_id].setXRange(0, int(TRAIN_CONF['epoch']), 0.05)
        self.insert_text("Start Training Model ...")
        self.worker = TAO_Train()
        self.worker.start()
        self.worker.trigger.connect(self.update_t2_train_log)
        self.ui.t2_bt_train.setEnabled(False)
        self.ui.t2_bt_stop.setEnabled(True)
    
    """ T2 -> 按下 stop 的事件 """
    def t2_stop_event(self):
        self.ui.t2_bt_train.setEnabled(True)
        self.ui.t2_bt_stop.setEnabled(False)
        self.worker.terminate()
        self.worker = None

    """ T2 -> 更新 eval 的內容 """
    def update_t2_eval_log(self, data):
        if data != "end":
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")                                # 插入內容
            self.mv_last_line()
            pass            
        else:
            self.insert_text("Evaluating Model ... Finished !!!")
            self.worker_eval.quit()
            self.swith_page_button(previous=1, next=1)

    """ T2 -> 更新 console 內容 """
    def update_t2_train_log(self, data):
        
        if bool(data):
            cur_epoch, avg_loss, val_loss, max_epoch = data['epoch'], data['avg_loss'], data['val_loss'], int(TRAIN_CONF['epoch'])
            
            log=""
            if val_loss is not None: 
                self.t2_var["val_epoch"].append(cur_epoch)
                self.t2_var["val_loss"].append(val_loss)
            else:
                self.t2_var["avg_epoch"].append(cur_epoch)
                self.t2_var["avg_loss"].append(avg_loss)
            
            if cur_epoch%5==0 and val_loss==None: 
                pass
            else:
                log = "{} {} {}\n".format(  f'[{cur_epoch:03}/{max_epoch:03}]',
                                            f'AVG_LOSS: {avg_loss:06.3f}',
                                            f'VAL_LOSS: {val_loss:06.3f}' if val_loss is not None else ' ')

                self.pws[self.current_page_id].clear()                                                  # 清除 Plot
                self.consoles[self.current_page_id].insertPlainText(log)                                # 插入內容
                self.mv_last_line()

                self.pws[self.current_page_id].plot(self.t2_var["avg_epoch"], self.t2_var["avg_loss"], pen=pg.mkPen('r', width=2), name="average loss")
                self.pws[self.current_page_id].plot(self.t2_var["val_epoch"], self.t2_var["val_loss"], pen=pg.mkPen('b', width=2), name="validation loss")
                self.update_progress(self.current_page_id, cur_epoch, max_epoch)
        else:
            self.insert_text("Training Model ... Finished !!!")
            self.ui.t2_bt_train.setEnabled(True)
            self.worker.quit()
            self.insert_text("Start Evaluating Model ...")
            self.worker_eval = TAO_VAL()
            self.worker_eval.start()
            self.worker_eval.trigger.connect(self.update_t2_eval_log)

    """ T2 -> 將 t2 的資訊映射到 TRAIN_CONF 上 """
    def update_train_conf(self):
        TRAIN_CONF['key'] = self.ui.t2_key.toPlainText()
        TRAIN_CONF['epoch'] = self.ui.t2_epoch.toPlainText()
        TRAIN_CONF['input_shape'] = self.ui.t2_input_shape.toPlainText()
        TRAIN_CONF['learning_rate'] = self.ui.t2_lr.toPlainText()
        TRAIN_CONF['output_name'] = self.ui.t2_model_name.toPlainText()
        TRAIN_CONF['batch_size'] = self.ui.t2_batch.toPlainText()
        # TRAIN_CONF['checkpoint'] = self.ui.t2_bt_checkpoint
        TRAIN_CONF['custom'] = self.ui.t2_c1.toPlainText()

    ####################################################################################################
    #                                                T1                                                #
    ####################################################################################################

    """ T1 -> 取得任務 並更新 模型清單 """
    def get_task(self):
        if self.ui.t1_combo_task.currentIndex()== -1:
            self.sel_idx[0]=0
        else:
            self.ui.t1_combo_model.clear()
            TRAIN_CONF['task'] = self.ui.t1_combo_task.currentText()
            self.ui.t1_combo_model.addItems(list(OPT[TRAIN_CONF['task']].keys()))
            self.ui.t1_combo_model.setCurrentIndex(-1)

            self.sel_idx[0]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

    """ T1 -> 取得模型 並更新 主幹清單 """
    def get_model(self):
        if self.ui.t1_combo_model.currentIndex()== -1:
            self.sel_idx[1]=0
        else:
            self.ui.t1_combo_bone.clear()
            TRAIN_CONF['model'] = self.ui.t1_combo_model.currentText()
            self.ui.t1_combo_bone.addItems(list(OPT[TRAIN_CONF['task']][TRAIN_CONF['model']])   )
            self.ui.t1_combo_bone.setCurrentIndex(-1)
            self.sel_idx[1]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

    """ T1 -> 取得主幹 並更新 層數清單 """
    def get_backbone(self):
        if self.ui.t1_combo_bone.currentIndex()== -1:
            self.sel_idx[2]=0
        else:
            self.ui.t1_combo_layer.clear()
            TRAIN_CONF['backbone'] = self.ui.t1_combo_bone.currentText()
            if TRAIN_CONF['backbone'] in ARCH_LAYER.keys():
                self.ui.t1_combo_layer.setEnabled(True)
                self.ui.t1_combo_layer.addItems(ARCH_LAYER[TRAIN_CONF['backbone']])
                self.ui.t1_combo_layer.setCurrentIndex(-1)
            self.sel_idx[2]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

    """ T1 -> 取得層數 """
    def get_nlayer(self):
        if self.ui.t1_combo_layer.currentIndex()== -1:
            self.sel_idx[3]=0
        else:
            TRAIN_CONF['nlayer'] = self.ui.t1_combo_layer.currentText()
            self.sel_idx[3]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects)) 


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = UI()
    window.show()
    sys.exit(app.exec_())