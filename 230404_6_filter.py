# -*- coding: utf-8 -*-
"""230404_6_filter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z1QP6hFqWqd_W56diwdDZ4pz42g1Poiz
"""

from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

(train_images, train_labels),(test_images, test_labels)= keras.datasets.mnist.load_data() 

image_size = 27
train_images = np.expand_dims(train_images, axis = -1)
train_images = tf.image.resize(train_images,[image_size,image_size])
train_images = np.squeeze(train_images)
train_images= train_images/255.0

test_images = np.expand_dims(test_images, axis = -1)
test_images = tf.image.resize(test_images,[image_size,image_size])
test_images = np.squeeze(test_images)
test_images= test_images/255.0

for k in range(int(train_images.shape[0])) :
      for i in range(image_size):
            for j in range(image_size):
                if train_images[k, i,j] <0.3:
                    train_images[k, i, j] = 0
                elif train_images[k, i,j] <0.7:
                    train_images[k, i,j] = 0.5
                else :
                    train_images[k, i,j] = 1

for k in range(int(test_images.shape[0])) :
  for i in range(image_size):
    for j in range(image_size):
      if test_images[k, i,j] <0.3:
        test_images[k, i, j] = 0
      elif test_images[k, i,j] <0.7:
        test_images[k, i,j] = 0.5
      else :
        test_images[k, i,j] = 1

train = [[],[],[],[],[],[],[],[],[],[]]
for k in range(train_images.shape[0]):
  if train_labels[k] ==0:
    train[0].append(train_images[k])
  elif train_labels[k] ==1:
    train[1].append(train_images[k])
  elif train_labels[k] ==2:
    train[2].append(train_images[k])
  elif train_labels[k] ==3:
    train[3].append(train_images[k])
  elif train_labels[k] ==4:
    train[4].append(train_images[k])
  elif train_labels[k] ==5:
    train[5].append(train_images[k])
  elif train_labels[k] ==6:
    train[6].append(train_images[k])
  elif train_labels[k] ==7:
    train[7].append(train_images[k])
  elif train_labels[k] ==8:
    train[8].append(train_images[k])
  elif train_labels[k] ==9:
    train[9].append(train_images[k])
  else:
    train_images[k] = train_images[k]

#6개 edge 선언
filters = [[],[],[],[],[],[]]
filters[0] = [[1,2,1],[0,0,0],[-1,-2,-1]]#0
filters[1] = [[0,1,3],[0,0,0],[-3,-1,0]] #30
filters[2] = [[0,0,3],[-1,0,1],[-3,0,0]] #60
filters[3] = [[-1,0,1],[-2,0,2],[-1,0,1]]#90
filters[4] = [[-3,0,0],[-1,0,1],[0,0,3]] #120
filters[5] = [[-3,-1,0],[0,0,0],[0,1,3]] #150

def flattening(image,filter):
  edge_sum = [[0]*(9) for _ in range(9)]
  flat_result_sum=[[0] for _ in range(81)]
  for i in range(0, image.shape[0]-2, 3):
    for j in range(0, image.shape[1]-2, 3):
      image_patch = image[i:i+3, j:j+3]
      edge_sum[int(i/3)][int(j/3)]=np.sum(np.multiply(image_patch,filter))
  
  flat_result_sum=[element for row in edge_sum for element in row]
  return flat_result_sum

flat_result_sum=[[],[],[],[],[],[]]
for i in range(6):
  flat_result_sum[i] = flattening(train_images[0],filters[i])

def edge_pattern(flattened_sum):
  
  temp = []
  max_index=[[0] for _ in range(81)]
  edge_pattern=[[0] for _ in range(80)]
  
    
  for i in range(81):
    temp.clear()
    for j in range(6):
      temp.append(flattened_sum[j][i])

    if(max(temp)==0):
      max_index[i] = 20
    elif(max(temp)==temp[0]):
      max_index[i] = 0
    elif(max(temp)==temp[1]):
      max_index[i] = 1
    elif(max(temp)==temp[2]):
      max_index[i] = 2
    elif(max(temp)==temp[3]):
      max_index[i] = 3
    elif(max(temp)==temp[4]):
      max_index[i] = 4
    elif(max(temp)==temp[5]):
      max_index[i] = 5
    else:
      max_index[i] = 20
      
  for i in range(80):
    if(max_index[i]==20):
      if(max_index[i+1]==20):
        edge_pattern[i]=0
      else:
        #edge_pattern[i]=5
        edge_pattern[i]=max_index[i+1]
    elif(max_index[i+1]==20):
      #edge_pattern[i]=5
      edge_pattern[i]=max_index[i]
    else:
      edge_pattern[i] = max_index[i]-max_index[i+1]
        
  return edge_pattern

def dtw_distance(signal1, signal2):
    # 시계열 데이터의 길이를 저장합니다.
    n, m = len(signal1), len(signal2)
    
    # DTW 행렬을 생성합니다.
    dtw = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dtw[i][0] = np.inf
    for i in range(1, m+1):
        dtw[0][i] = np.inf
    dtw[0][0] = 0
    
    # DTW 행렬을 계산합니다.
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(signal1[i-1] - signal2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    
    # 최종적인 DTW 거리를 반환합니다.
    return dtw[-1][-1]

import time
from tqdm import tqdm
import pandas as pd

# test_images 를 모두 1차원 벡터화한다. 
train_images_flat = [[] for _ in range(10)]
temp_1 = [[],[],[],[],[],[]]
temp_2 = [[] for _ in range(80)]

for k in range(10):
  for i in range(101):
      for j in range(6):
          temp_1[j] = flattening(train[k][i],filters[j])
      temp_2 = edge_pattern(temp_1)
      train_images_flat[k].append(temp_2)

# 1. train 이미지를 0부터 9까지 101개씩 평면화하기.
# 2. 0번 이미지 와 나머지 1~100번 이미지 까지 DTW 구해서 평균값을 구해서 표로 만들기

dtw_d = np.zeros([10,10])
temp = 0

for i in tqdm(range(10)):
  for j in range(10):
    sum = 0
    for k in range(100):
      sum += dtw_distance(train_images_flat[i][0],train_images_flat[j][k])
    temp = sum/100
    dtw_d[i][j] = temp

df = pd.DataFrame (dtw_d, columns = ['0','1','2','3','4','5','6','7','8','9'])
df

dtw_d = np.zeros([10,10])
temp = 0

for i in tqdm(range(10)):
  for j in range(10):
    sum = 0
    for k in range(100):
      sum += dtw_distance(train_images_flat[i][11],train_images_flat[j][k])
    temp = sum/100
    dtw_d[i][j] = temp

df = pd.DataFrame (dtw_d, columns = ['0','1','2','3','4','5','6','7','8','9'])
df