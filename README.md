## TFModelQuantizer

A package which converts saved models weights to quantized FP32 ,FP16 and INT8 for faster Inference through TF-TensorRT. A demonstration of the package on Resnet-50 is provided in the [Colab Notebook](https://colab.research.google.com/drive/1CDKGzLtt2Zy51TWt4bRQou3i7Mvy_THa?usp=sharing)

A detailed overview of the package and its optimizations have been provided in the [Kaggle Notebook](https://www.kaggle.com/abhilash1910/tfmodelquantizer-quantization-of-tf-models/)

<img src="https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/graphics/tf-trt-workflow.png">

## Package Specifications

The TFModelQuantizer uses the TensorRT (TRT) Inference for faster performance. The package can be found in [Github](https://github.com/abhilash1910/TFModelQuantizer/) and can be installed as follows:

```!pip install TFModelQuantizer```

In the next step, we load a model from the Keras.applications module. For instance , Resnet 50 has been chosen for this purpose. Since the package uses TRT dynamic runtime for inference, we will only be performing inference on this. The beabuty of dynamic runtime during model conversion is that there is less memory overhead. Previously during conversion , static mode was used. Only subgraph segmentation happens before runtime , and the optimization part of the subgraph happens during the runtime. 
We first load the Resnet 50 model  and save it:

```python
model = ResNet50(weights='imagenet')
model_dir = 'savedmodels/resnet50_saved_model'
model = ResNet50(include_top=True, weights='imagenet')
model.save(model_dir)
```

Now we call the TFModelQuantizer class from the package. Then we specify the quantization precision (either of FP32,FP16 or INT8 with calibration). This takes as input the saved model from the directory and the precision and converts the model weights into the new precision and freezes the graph. Essentially this means it converts the model weights to a ".pb" format (protobuffer).This is done by the following lines:

```python
model_quantizer_fp16=TFModelQuantizer.TFModelQuantizer(model_dir,"FP16")
model_q_16=model_quantizer_fp16.quantize(model_dir+'_FP16')
```

For instance in the example here, we are converting to FP16, which will provide a significant boost of speed in runtime.
To check the runtime of the inference of the elephant image, we can do:

```python
start_time=time.time()
preds = model_q_16.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print("Inference Time on FP16 Quantized Model",time.time()-start_time)
```
And thus we get the outputs (inference) and the time which is significantly smaller than the unquantized model. When iterated over optimized TF datasets like TFrecords , this package can provide significant speedupds, with just 3 lines of code. The following inference sample is taken from [Keras official site](https://keras.io/api/applications/)


## Comparatibe Analysis

An image of the inference performance on quantization is provided below. INT8 (calibrated) is fastest followed by FP16 and FP32.

![img1](images/quantization_stats.PNG)


## Further Work

An additional performance analysis with respect to larger models from keras would be done. Future plan is to focus on distributed inference.


## License

MIT
