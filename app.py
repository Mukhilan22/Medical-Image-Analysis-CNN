import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import keras
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.utils import load_img, img_to_array

# @st.cache(allow_output_mutation=True)

# LOADING CNN MODELS
fracture=keras.models.load_model("fracture.h5")
tumor=keras.models.load_model("tumor.h5")
pnem=keras.models.load_model("pnem.h5")
skc=keras.models.load_model("best_model.h5")

st.title(""" MEDICAL IMAGE DIAGNOSIS :medical_symbol:""")
st.markdown("""---""")

st.write(""" ## FRACTURE DIAGNOSIS 	:bone:""")

fracimg = st.file_uploader("Please upload a XRay image", type=["jpeg","jpg", "png"],key = "fracture")

def fracture_predict(image_data, model):

      size = (224,224)    
      image = ImageOps.fit(image_data, size)
      imag2 = img_to_array(image)
      imaga2 = np.expand_dims(imag2,axis=0) 
      ypred = fracture.predict(imaga2)
        
      return ypred

if fracimg is None:
    st.text("Please upload an image file")
else:
    image = Image.open(fracimg)
    st.image(image, width=350)
    predictions = fracture_predict(image, fracture)
    
    a=predictions[0]
    st.write(a)
    if a<0.5:
          st.write(""" ### FRACTURE IS PREDICTED """)   
    else:
          st.write(""" ### NO FRACTURE IS PREDICTED """)


st.markdown("""---""")
st.write(""" ## BRAIN TUMOR DIAGNOSIS :brain:""")

tumorimg = st.file_uploader("Please upload a MRI scan", type=["jpeg","jpg", "png"],key = "tumor")

labels = ['GLIOMA TUMOR','MENINGIOMA TUMOR','NO TUMOR','PITUTIARY']

def tumor_predict(image_data, model):

      size = (150,150)    
      image = ImageOps.fit(image_data, size)
      img_array = np.array(image)
      img_array = img_array.reshape(1,150,150,3)

      ypred = model.predict(img_array)
        
      return ypred

if tumorimg is None:
    st.text("Please upload an image file")
else:
    image = Image.open(tumorimg)
    st.image(image, width=350)
    predictions = tumor_predict(image, tumor)

    st.write(predictions)
    
    index = predictions.argmax()

    st.write(labels[index])
    st.write("### THE PATIENT HAS : " + labels[index])

st.markdown("""---""")
st.write(""" ## PENUMONIA DIAGNOSIS :lungs:""")

pnemimg = st.file_uploader("Please upload a Chest XRay image", type=["jpeg","jpg", "png"],key = "pneumonia")

def pneumonia_predict(image_data, model):

      size = (120,120)    
      img1 = ImageOps.fit(image_data, size)
      imag1 = img_to_array(img1)
      imaga1 = np.expand_dims(imag1,axis=0) 
      imaga1 = np.concatenate([imaga1] * 3, axis=-1)

      ypred = model.predict(imaga1)
        
      return ypred 

if pnemimg is None:
    st.text("Please upload an image file")
else:
    image = Image.open(pnemimg)
    st.image(image, width=350)
    predictions = pneumonia_predict(image, pnem)
    
    a=predictions[0]
    st.write(a)
    if a>0.5:
          st.write(""" ### PNEUMONIA IS PREDICTED """)   
    else:
          st.write(""" ### PATIENT HAS NO PNEUMONIA """)

st.markdown("""---""")
st.write(""" ## SKIN CANCER DIAGNOSIS :large_yellow_square:""")

skcimg = st.file_uploader("Please upload a Skin Cancer scan", type=["jpeg","jpg", "png"],key = "skc")

classes = {0: ('AKIEC - Actinic keratoses and intraepithelial carcinomae'),  
	     1: ('BCC - Basal Cell Carcinoma'),  
           2: ('BKL - Benign Keratosis-like Lesions'),  
           3: ('DF - Dermatofibroma'), 
           4: ('NV - Melanocytic Nevi'),  
           5: ('VASC - Pyogenic Granulomas and Hemorrhage'), 
           6: ('MEL - Melanoma')}

def skincancer_predict(image_data, model):

      size = (28,28)    
      image = ImageOps.fit(image_data, size)
      img_array = np.array(image)
      img_array = img_array.reshape(1,28,28,3)

      ypred = model.predict(img_array)
        
      return ypred

if skcimg is None:
    st.text("Please upload an image file")
else:
    image = Image.open(skcimg)
    st.image(image, width=350)
    predictions = skincancer_predict(image, skc)

    st.write(predictions)
    
    max_prob = max(predictions[0])
    class_ind = list(predictions[0]).index(max_prob)
    class_name = classes[class_ind]
    st.write(class_name)

    st.write("### THE PATIENT HAS : " + class_name)
