import streamlit as st
import pandas as pd
import numpy as np
import cv2 
import script as sr
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import easydict
from imutils import paths
import re

# 페이지 이름 지정
st.set_page_config(
     page_title="🌲 Pine wilt disease Web 🌲",
#      layout="wide",
     initial_sidebar_state="expanded"
 )
 # 사이드바
# st.sidebar.header("소나무재선충병")
# name = st.sidebar.selectbox("Menu", ['웹캠'])

# 로고 이미지 넣기
st.image('./title.jpg')
st.write("""
## Pine Tree Wilt Disease Calc Web
소나무 재선충병 발생 면적을 계산해 보세요!
***
""")

# 1. 사진 계산
st.header("## 피해 사진 넣기")
photo = st.file_uploader("사진을 업로드해 주세요.", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# 1-1. 이미지 저장
for i in photo :
    sr.save_uploaded_file('photo', i)
    
# 1-2. 이미지 전처리
HEIGHT, WIDTH, CHANNEL = 512, 512, 3
X_test = sr.imagePrep('./photo/*', WIDTH, HEIGHT, CHANNEL)

# 1-3. 원본과 전처리 후 사진 출력
model_path= './model/test (528).hdf5'
ae_imgs = sr.test_model(model_path, X_test) 
img = sr.display_mask_test(ae_imgs)
st.pyplot(img.all()) 
# showPyplotGlobalUse = false # 스트림릿 경고문 없애기

# 1-4. 전처리 후 사진 저장하기
file_list = os.listdir('./photo')
for i in range(len(file_list)-1): 
    sr.display_mask_save(img, i) # 0.jpg 부터 시작함

# 좌표
img_list = os.listdir('./testtarget')
for i in range(len(img_list)-1):
    st.write(f'#### {i+1}번째 사진')
    st.write('- 피해 좌표')
    img = cv2.pyrDown(cv2.imread(f'./testtarget/{i}.jpg', cv2.IMREAD_UNCHANGED))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold image
    ret, threshed_img = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arr = np.empty((0,2), int)
    for c in contours:
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        arr = np.append(arr, box, axis = 0)

        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
        #arr = np.append(arr, np.array(box), axis = 0)
        #arr = np.append(arr, np.array(box))

    #     print(len(contours))
    
    arr_split = np.split(arr, len(contours))
    arr_mean = []

    for x in arr_split:
        arr_mean = np.append(arr_mean, x.mean(axis=0))

    arr_mean = np.split(arr_mean, len(contours))
    arr_mean = np.int0(arr_mean)
    
    color = (255,0,0)

    for mean in arr_mean:
        x = mean[0]
        y = mean[1]
        cv2.line(img, (x,y), (x,y), color, 5)
    arr_mean = pd.DataFrame(arr_mean, columns=['x좌표', 'y좌표']) 
    arr_mean.index = arr_mean.index+1
    arr_mean
#     cv2.imwrite(f"coordinate/{i+1}.png", img)   
#     cv2.imshow("contours", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    
# 면적
    x, img = sr.display_mask(ae_imgs, i)
    ds_S, st_S = sr.calArea(x) # 계산
    st.write(f" - 질병면적 : {ds_S[i]:.2f}, 정상면적 : {st_S[i]:.2f} \n- 질병면적의 비율 : ", round(ds_S[i]/st_S[i], 2))
