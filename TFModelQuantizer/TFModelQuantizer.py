import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
import numpy as np

precision_vals={"FP32":tf_trt.TrtPrecisionMode.FP32,"FP16":tf_trt.TrtPrecisionMode.FP16,"INT8":tf_trt.TrtPrecisionMode.INT8}

class Optimizer():
  def __init__(self,saved_model_dir=None):
    self.loaded_model=None
    try:
      if saved_model_dir!=None:
        self.load_model(saved_model_dir)
    except:
      raise(Exception("Could not access Saved Model Configurations"))
  def load_model(self,input_dir):
    saved_loaded_model=tf.saved_model.load(input_dir,tags=[tag_constants.SERVING])
    wrapper_fp32=saved_loaded_model.signatures["serving_default"]
    self.loaded_model=wrapper_fp32

  def predict(self,inputs):
    if self.loaded_model is None:
      raise(Exception("Model not loaded"))
    x=tf.constant(inputs.astype('float32'))
    y=self.loaded_model(x)
    try:
      pred_columns=['predictions','probs']
      if pred_columns[0] in y:
        final_preds=y['predictions'].numpy()
      elif pred_columns[1] in y:
        final_preds=y['probs'].numpy()
      else:
        final_preds=y[next(iter(y.keys()))]

    except:
        raise(Exception("Unable to get predictions from model"))
    return final_preds
  
class TFModelQuantizer():
  def __init__(self,input_dir,precision,calibration=None):
    try:
      if input_dir!=None:
        self.input_dir=input_dir
    except:
      raise(Exception("Could not access input_dir"))
      
    self.precision=precision
    self.calibration=None
    self.model_load=None
    if not calibration is None:
      self.calibration_converter(calibration)
  
  def calibration_converter(self,calibration):  
    def calibration_convert():
      yield (calibration.astype('float32'),)
      #yield (tf.constant(calibration.astype('float32')),)
    self.calibration=calibration_convert

    
  def quantize(self,output_dir,max_workspace_size_bytes=(1<<32),**kwargs):
    if self.precision=="INT8" and self.calibration is None:
      raise(Exception("INT8 needs to be calibrated"))
    trt_precision=precision_vals[self.precision]
    quantizer_params=tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision, 
                                                            max_workspace_size_bytes=max_workspace_size_bytes,
                                                            use_calibration= self.precision == "INT8")
    quantizer_params=quantizer_params._replace(maximum_cached_engines=100)
    quantizer_params=quantizer_params._replace(minimum_segment_size=3)
    quantizer=tf_trt.TrtGraphConverterV2(input_saved_model_dir=self.input_dir,
                                conversion_params=quantizer_params)
    if self.precision=="INT8":
      quantizer.convert(calibration_input_fn=self.calibration)
    else:
      quantizer.convert()
    quantizer.save(output_saved_model_dir=output_dir)
    
    return Optimizer(output_dir)

  def inference(self,input_data):
    return self.model_load.predict(input_data)




 
    