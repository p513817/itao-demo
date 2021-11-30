"""
####################################################################################################
#                                               CONF                                               #
####################################################################################################
"""

OPT = {
    "Image Classification":{
        "ResNet":{"ResNet"}, 
        "VGG":{"VGG"}, 
        "GoogleNet":{"GoogleNet"}, 
        "AlexNet":{"AlexNet"}
    },
    "Objected Detection":{
        "DetectNet_v2":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        "SSD":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        "YOLOv3":{"ResNet", "VGG", "GoogleNet", "AlexNet", "CSPDarknet53", "CSPDarknet101"}, 
        "YOLOv4":{"ResNet", "VGG", "GoogleNet", "AlexNet", "CSPDarknet53", "CSPDarknet101"}, 
    },
    "Segmentation":{
        "ResNet":{"ResNet"}, 
        "VGG":{"VGG"}, 
        "GoogleNet":{"GoogleNet"}, 
        "AlexNet":{"AlexNet"}
    },
    "Other":{
        None
    }
}

ARCH_LAYER= {
    "ResNet":["10", "18", "50"],
    "VGG":["16", "19"],
    "GoogleNet":["Default"],
    "AlexNet":["Default"],
    "CSPDarknet53":["Default"],
    "CSPDarknet101":["Default"]
}

TRAIN_CONF = {
    "key":"nvidia_tlt",
    "task":"",
    "model":"",
    "backbone":"",
    "nlayer":"",
    "dataset_path":"",
    "label_path":"",
    "checkpoint":"",
    "output_name":"",
    "input_shape":"",
    "epoch":"",
    "batch_size":"",
    "learning_rate":"",
    "custom":""
}

PRUNE_CONF = {
    "key":"",
    "input_model":"",
    "thres":"",
    "output_name":""
}

RETRAIN_CONF = {
    "key":"",
    "epoch":"",
    "learning_rate":"",
    "output_name":"",
    "batch_size":"",
    "custom":""
}
