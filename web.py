from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image ,  ImageOps
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import glob
import tensorflow as tf
import script as sr

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm


# 모델 불러오기
model = load_model('./test (217).hdf5')

# 학습된 결과를 출력
def test_model(path, X_test):
    model = load_model(path)
    ae_imgs = model.predict(X_test) 

    # 출력될 이미지의 크기
    fig, ax = plt.subplots((ae_imgs.shape[0]//5+1)*2, 5, figsize=(16,16)) 
    
    # 이미지를 차례대로 나열
    for i, (src, dst) in enumerate(zip(X_test,ae_imgs)): # src=원본, dst=예측 후
        r = i//5 # 행의 인덱스
        c = i%5 # 열 인덱스
        
        # 테스트 이미지 출력
        ax[2*r,c].imshow(src) 
        ax[2*r,c].axis('off')
        
        # 오토인코딩 결과(예측 이미지) 출력
        ax[2*r+1,c].imshow(dst, cmap=cm.Greys) 
        ax[2*r+1,c].axis('off')
        
    return ae_imgs, fig

def display_mask(predicted , i):
    model = np.round(predicted)
    x = np.array(model[i].reshape(512,512) *255).astype('uint8')
    img = Image.fromarray(x)
#     img.show()
    return x, img



        
# 업로드 된 이미지 저장하는 함수    
# 디렉토리와 파일 주면, 해당 디렉토리에 파일 저장
def save_uploaded_file(directory, file) :
    if not os.path.exists(directory): #  디렉토리 없으면 만들기
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), 'wb') as f: # 파일 저장하기
        f.write(file.getbuffer())
#     return st.success('Saved file : {} in {}' .format(file.name, directory))


## 페이지 이름 지정
st.set_page_config(
     page_title="🌲 Pine wilt disease Web 🌲",
     layout="wide",
     initial_sidebar_state="expanded"
 )

## 로고 이미지 넣기
st.image("./title.png", use_column_width=True)


## 사이드바
data = pd.read_csv("pine.csv") # 전국 고사목 수
data = data.set_index("년도") # 년도 데이터를 인덱스로 지정
chart_data = pd.DataFrame(data)

st.sidebar.header("소나무재선충병")
name = st.sidebar.selectbox("Menu", ['개요', '면적 추출 예시', '면적 계산하기'])

# 1) 사이드바 1번 - 전국 고사목 수
if name == "개요":
    st.write("### 🌳 소나무재선충병 피해 본수와 투입 예산")
    st.image("./그림2.jpg", width=500)
    st.write("""
    ### 🌳 소나무재선충병이란?
    - 매개충인 하늘소에 의해 빠른 확산이 이루어짐
    - 상처 부위를 통해 침입한 재선충이 소나무의 수분·양분의 이동통로를 막아 나무를 죽게 하는 병
    - 치료약이 없어 감염되면 100% 고사
    - 지금까지 재선충에 의해 고사한 소나무류는 총 1,200만 본
    - 22년 4월 말 기준 피해목이 작년과 비교해 22.6% 증가
    """)

    st.write("### 🌳 전국 고사목 수(~2020년)")
    st.bar_chart(chart_data)
    # 2022년 현황
    st.write("### 🌳 2022년도 현황")
    col1, col2 = st.columns(2)
    col1.metric(label="발생 시·군·구", value="135개", delta="4개 지역")
    col2.metric(label="고사목 수", value="38만 본", delta="22.6%")

# 2) 사이드바 2번 - 면적 추출 예시    
if name =="면적 추출 예시":
    st.markdown("#### 🌳 소나무재선충병 이미지 처리 예시")
    st.write("- 산림 모형과 전처리 이미지")
 
    st.image("./그림.png", use_column_width=True )

# 3) 사이드바 3번 - 면적 계산하기    
# 사진 업로드(여러장도 가능하게)        
if name =="면적 계산하기":
    st.subheader("🌳 면적 계산하기")

    uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True,
                                  type=["png","jpg","jpeg"])
    for i in uploaded_files :
    # 업로드 된 사진 저장하기
        save_uploaded_file('uploaded_files', i)
    
    
    # 저장된 사진에 대하여..
    # 1) 이미지 전처리
    HEIGHT, WIDTH, CHANNEL = 512, 512, 3
    X_test = sr.imagePrep('./uploaded_files/*.jpg', WIDTH, HEIGHT, CHANNEL)  
  
    
    # 2) 원본과 전처리 후 사진 보여주기 
    model_path = './test (217).hdf5'
    y_pred, fig = test_model(model_path, X_test)
    st.pyplot(fig, figsize=(1,1))
    
    
    # 3) 모든 사진에 대한 면적 출력
    count = os.listdir('./uploaded_files')
    for i in range(len(count)) :
        x, img = display_mask(y_pred, i)
        ds_S, st_S = sr.calArea(x) # 계산
        st.write(f"#### • {i+1}번째 사진")
        st.image(img, width=300)
        st.write(f"##### - 질병면적 : {ds_S[0]:.2f}, 정상면적 : {st_S[0]:.2f}")
    

    
    

    



    

    
