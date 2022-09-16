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


# 이미지 전처리
def imagePrep(path_pattern, WIDTH, HEIGHT, CHANNEL): 
    
    filelist = glob.glob(path_pattern)
    fileabslist = [os.path.abspath(fpath) for fpath in filelist]
    X = []
    for fname in fileabslist:
        img = cv2.imread(fname).astype('float32') / 255
        #print(img.shape)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        if CHANNEL == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = np.expand_dims(img, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        X.append(img)
    X_array = np.array(X)
    return X_array

#비교 테스트 모델, 학습된 모델 input
def test_model(path ,test_data):
    model = load_model(path)
    #앞서 만든 오토인코더 모델에 집어 넣습니다.
    ae_imgs = model.predict(test_data)           
    random_test = np.random.randint(test_data.shape[0], size=5) 
    #출력될 이미지의 크기를 정합니다.
    plt.figure(figsize=(70, 20))   
    #랜덤하게 뽑은 이미지를 차례로 나열합니다.
    for i, image_idx in enumerate(random_test):
        ax = plt.subplot(2, 7, i + 1) 
        #테스트할 이미지를 먼저 그대로 보여줍니다.
        plt.imshow((test_data[image_idx]*255).astype('uint8'), cmap=cm.Greys) 
        ax.axis('off')
        ax = plt.subplot(2, 7, 7 + i +1)
        #오토인코딩 결과를 다음열에 출력합니다.
        plt.imshow((ae_imgs[image_idx]*255).astype('uint8'), cmap=cm.Greys)  
        ax.axis('off')      
#         plt.show()
    return ae_imgs

def display_mask_test(model , i , test_data):
    model = np.round(model)
    x = np.array(model[i].reshape(512,512,3) *255).astype('uint8')
    img = Image.fromarray(x)
    #img.show()
    #이미지 저장.
    #img.save(f"./testtarget/{i}.jpg")
    plt.figure(figsize=(70, 20))
    # 원본
    ax = plt.subplot(2, 7, 1) 
    plt.imshow((test_data[i]*255).astype('uint8'))
    ax.axis('off')
    # 전처리 후
    ax = plt.subplot(2, 7, 2) 
    plt.imshow(img)
    ax.axis('off')
    return model

def display_mask_save(model , i):
    model = np.round(model)
    x = np.array(model[i].reshape(512,512,3) *255).astype('uint8')
    img = Image.fromarray(x)
    #img.show()
    #이미지 저장.
    img.save(f"./testtarget/{i}.jpg")
    plt.imshow(img)
    return model

# 업로드 된 이미지 저장하는 함수    
def save_uploaded_file(directory, file) :
    #  디렉토리 없으면 만들기
    if not os.path.exists(directory): 
        os.makedirs(directory)
    # 파일 저장하기
    with open(os.path.join(directory, file.name), 'wb') as f: 
        f.write(file.getbuffer())
#     return st.success('Saved file : {} in {}' .format(file.name, directory))


## 사이드바
# data = pd.read_csv("pine.csv") # 전국 고사목 수
# data = data.set_index("년도") # 년도 데이터를 인덱스로 지정
# chart_data = pd.DataFrame(data)

st.sidebar.header("소나무재선충병")
name = st.sidebar.selectbox("Menu", ['사진', '웹캠'])



# 1. 사진 계산기
if name=='사진':
    st.subheader("🌳 사진 발병 면적 계산")
    uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True, type=["jpg","png"])
 
    # 1-1. 업로드 된 사진 저장하기
    for i in uploaded_files :
        st.image(i)
        print(i)
        save_uploaded_file('uploaded_files', i)        

        # 1-2. 이미지 전처리
        HEIGHT, WIDTH, CHANNEL = 512, 512, 3
        X_test = imagePrep('./uploaded_files/*.png', WIDTH, HEIGHT, CHANNEL)

        # 1-3. 원본과 전처리 후 사진 보여주기
        model_path= './model/test (528).hdf5'
        ae_imgs = test_model(model_path, X_test) 
        img = display_mask_test(ae_imgs, i, X_test)
        
        
        
        # 전처리 이미지 저장
        

        
       # 1-4. 좌표 출력   
    
        

#            ae_imgs.save("img.png") -> numpy.ndarray has no attribute save
#             cv2.imwrite('./img.png',ae_imgs) # 아무것도 안 뜸(저장 확인 불가,,)

            
            
# 안됨..            
#             for i in range(0, 5):
#                 plt.savefig('./ae_imgs/img%d.png'%i)

#         # 실패..
# #         1-4. 좌표 출력
# #         ae_imgs 저장하기
#             st.write(ae_imgs.shape)
#             st.write(type(ae_imgs))
#             ae_imgs = np.asarray(ae_imgs, dtype=int)
#             st.write(type(ae_imgs))
#             ae_imgs = np.reshape(512, 512, 3)
#     #       im = Image.fromarray(ae_imgs)
#     #         im = Image.fromarray(ae_imgs, 'RGB')
#             im.save("img.png")
#         img = cv2.pyrDown(cv2.imread('./img.png', cv2.IMREAD_UNCHANGED))
#         imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    
    
# # 2. webcam 사용 면적 계산
# if name=='웹캠':
#     from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
#     import av #strealing video library

#     st.title('Webcam Test')
#     st.write('Pine Tree')
#     webrtc_streamer(key='key')

#     # 2-1. webcam 영상 캡쳐하기
#     # 0=디폴트 카메라 디바이스를 의미, 0 대신 동영상 파일 위치 입력도 가능
#     webcam = cv2.VideoCapture(0) 
#     # 잘 연결되어 있지 않다면 종료
#     if not webcam.isOpened():
#         print("웹캠을 열 수 없습니다.")
#         exit()
#     # 연결되어 있으면 webcam.read() 통해 읽기    
#     while webcam.isOpened():
#         status, frame = webcam.read()
#     # status가 True이면 test라는 창에 캡처된 프레임 보여주기    
#         if status:
#             cv2.imshow("test", frame)
#     # 사용자가 키보드로 q 입력시 반복문 탈출
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     # 웹캠 연결 끊기        
#     webcam.release()
#     # 웹캠 영상 보여주기 위해 생성한 창 없애기
#     cv2.destroyAllWindows()

           
