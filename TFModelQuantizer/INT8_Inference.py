
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
dummy_one_batch = np.ones((32, 224, 224, 3),dtype='float32')

start_time=time.time()
model_quantizer_int8=TFModelQuantizer.TFModelQuantizer(model_dir,"INT8",dummy_one_batch)
model_q_8=model_quantizer_int8.quantize(model_dir+'_INT8')
diff_int8=time.time()-start_time
print(diff_int8)



