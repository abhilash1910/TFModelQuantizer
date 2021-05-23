# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:02:43 2020

@author: Abhilash
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:45:49 2020

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'TFModelQuantizer',         
  packages = ['TFModelQuantizer'],   
  version = '0.1',       
  license='MIT',        
  description = 'A Tensorflow TensorRT Model Quantizer for FP32 ,FP16 and calibrated INT8 model quantization.Runs on TensorRT default calibration on frozen model graphs for faster inference',   
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/TFModelQuantizer',   
  download_url = 'https://github.com/abhilash1910/TFModelQuantizer/archive/v_01.tar.gz',    
  keywords = ['Tensorflow Quantizer','TensorRT Inference','FP32','FP16','INT8','GraphConverter','Calibrated Inference'],   
  install_requires=[           

          'numpy',         
          'tensorflow',
          
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
