# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:46:50 2021

@author: Abhilash
"""

from tensorflow.keras.applications import ResNet50
import numpy as np
import TFModelQuantizer
import time
import h5py

model_dir = 'tmp_savedmodels/resnet50_saved_model'
model = ResNet50(include_top=True, weights='imagenet')
model.save(model_dir)
BATCH_SIZE = 32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3))

start_time=time.time()
model_quantizer_fp32=TFModelQuantizer.TFModelQuantizer(model_dir,"FP32")
model_q_32=model_quantizer_fp32.quantize(model_dir+'_FP32')
diff_fp32=time.time()-start_time
print(diff_fp32)



