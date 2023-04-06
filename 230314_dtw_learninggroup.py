#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


#6개 edge 선언
filters = [[],[],[],[],[],[]]
filters[0] = [[1,2,1],[0,0,0],[-1,-2,-1]]#0
filters[1] = [[0,1,3],[0,0,0],[-3,-1,0]] #30
filters[2] = [[0,0,3],[-1,0,1],[-3,0,0]] #60
filters[3] = [[-1,0,1],[-2,0,2],[-1,0,1]]#90
filters[4] = [[-3,0,0],[-1,0,1],[0,1,3]] #120
filters[5] = [[-3,-1,0],[0,0,0],[0,1,3]] #150


# In[4]:


def flattening(image,filter):
  edge_sum = [[0]*(9) for _ in range(9)]
  flat_result_sum=[[0] for _ in range(81)]
  for i in range(0, image.shape[0]-2, 3):
    for j in range(0, image.shape[1]-2, 3):
      image_patch = image[i:i+3, j:j+3]
      edge_sum[int(i/3)][int(j/3)]=np.sum(np.multiply(image_patch,filter))
  
  flat_result_sum=[element for row in edge_sum for element in row]
  return flat_result_sum


# In[5]:


flat_result_sum=[[],[],[],[],[],[]]
for i in range(6):
  flat_result_sum[i] = flattening(train_images[0],filters[i])


# In[6]:


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
        edge_pattern[i]=0
    elif(max_index[i+1]==20):
      #edge_pattern[i]=5
      edge_pattern[i]=0
    else:
      edge_pattern[i] = max_index[i]-max_index[i+1]
        
  return edge_pattern


# In[7]:


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


# In[8]:


from random import *
class chip_random:
    def __init__(self, size, threshold1 = 0.9, threshold2 = 0.95):
        '''
        size = memory input size 
        threshold1 = don't care  역치 //  0.9
        threshold2 = 1 역치 // 0.95
        '''
        self.size = size
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.cam = []
        self.counter = []

        for i in range(2**size-1):
            a = randint(1,2**size-1)
            while a in self.cam:
                a = randint(1,2**size-1)
            self.cam.append(a)
            self.counter.append(0)

    def reset(self):
        self.cam = []
        self.counter = []

        for i in range(2**self.size-1):
            a = randint(1,2**self.size-1)
            while a in self.cam:
                a = randint(1,2**self.size-1)
            self.cam.append(a)
            self.counter.append(0)

    def pattern_to_number(self, pattern):
        num = 0
        result = []
        for i in range(len(pattern)):
            if pattern[i]==1:
                num = num + 2**(len(pattern)-1-i)
        result.append(num)
        for i in range(len(pattern)):
            if pattern[i]==0.5:
                for k in range(len(result)):
                    result.append(result[k]+2**(len(pattern)-1-i))
    
        return result

    def change_threshold(self, threshold1, threshold2):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def learn(self, pattern):
        num = self.pattern_to_number(pattern)
        for i in range(len(self.cam)):
            if num.count(self.cam[i])!=0:
                self.counter[i] = self.counter[i] +1

        for i in range(len(num)):
            for j in range(len(self.cam)-1):
                if self.counter[j] > self.counter[j+1]:
                    temp =self.counter[j]
                    self.counter[j] = self.counter[j+1]
                    self.counter[j+1] = temp
                    temp =self.cam[j]
                    self.cam[j] = self.cam[j+1]
                    self.cam[j+1] = temp
    
    def address_up(self, pattern):
        num = self.pattern_to_number(pattern)
        for i in range(len(self.cam)-2,0,-1):
            if self.cam[i] in num:
                temp = self.cam[i]
                self.cam[i] = self.cam[i+1]
                self.cam[i+1] = temp

    def address_down(self, pattern):
        num = self.pattern_to_number(pattern)
        for i in range(len(self.cam)-2):
            if self.cam[i] in num:
                temp = self.cam[i]
                self.cam[i] = self.cam[i-1]
                self.cam[i-1] = temp


    def get_address(self, pattern):
        num = self.pattern_to_number(pattern)
        result = []
        for i in range(len(num)):
            if num[i] ==0:
                result.append(0)
            else :
                result.append(self.cam.index(num[i]))
        return result
  
    def get_binary(self, pattern):
        result = self.get_address(pattern)
        address = max(result)
        percent = address/len(self.cam)

        if percent < self.threshold1 :
            return 0.0
        elif percent <self.threshold2:
            return 0.5
        else:
            return 1.0


# In[9]:


class layer_random:
    def __init__(self, window_size, layer_size, stride = 0, threshold1 = 0.9, threshold2 = 0.95):
        self.L = [[0]*layer_size for _ in range(layer_size)]
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        for i in range(layer_size):
            for j in range(layer_size):
                self.L[i][j] = chip_random(window_size*window_size, self.threshold1, self.threshold2)
        
        if stride ==0:
            self.stride = window_size
        else :
            self.stride = stride
        self.window_size = window_size
        self.layer_size = layer_size

    def reset(self):
        '''
        layer가 초기화됨 내부 chip의 배열도 초기화
        '''
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                self.L[i][j].reset()

    def change_threshold(self, threshold1, threshold2):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                self.L[i][j].change_threshold(threshold1, threshold2)

    def learn(self, image):
        '''
        이미지 하나를 layer가 입력받게 되는 함수
        '''
        # 학습 가능한 이미지 사이즈인지 확인하는 코드 추가 작성
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                pattern= []
                for a in range(self.window_size):
                    for b in range(self.window_size):
                        pattern.append(image[self.stride*i+a][self.stride*j+b])
                self.L[i][j].learn(pattern)
    
    def address_up(self, image):
        '''
        이미지 안에 있는 패턴을 주소 1 up
        '''
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                pattern = []
                for a in range(self.window_size):
                    for b in range(self.window_size):
                        pattern.append(image[self.stride*i+a][self.stride*j+b])
                self.L[i][j].address_up(pattern)
            
    def address_down(self, image):
        '''
        이미지 안에 있는 패턴을 주소 1 up
        '''
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                pattern = []
                for a in range(self.window_size):
                    for b in range(self.window_size):
                        pattern.append(image[self.stride*i+a][self.stride*j+b])
                self.L[i][j].address_down(pattern)
    
    def out(self, image):
        '''
        이미지 넣고 출력되는 layer out 보여줌
        '''
        # 입력 가능한 이미지 사이즈인지 확인하는 코드 추가 작성
        out_image = [[0] * self.layer_size for _ in range(self.layer_size)]
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                pattern=[]
                for a in range(self.window_size):
                    for b in range(self.window_size):
                        pattern.append(image[self.stride*i +a][self.stride*j +b])
                out_image[i][j] = self.L[i][j].get_binary(pattern)
        return out_image

    def out_sum(self, image):
        '''
        이미지 넣고 출력되는 이미지의 전체 합 출력
        '''
        out_image = self.out(image)
        result = 0
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                result = result + out_image[i][j]
        return result
    
    def weighted_outsum (self, image):
        out_image = self.out(image)
        result = 0
        weight1 = 0
        weight2 = 0
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                if (i>self.layer_size/3)*(i<2*self.layer_size/3) : weight1 = 2
                else : weight1 = 1

                if(j>self.layer_size/3)*(j<2*self.layer_size/3) : weight2 = 2
                else : weight2 = 1

                result = result + weight1 * weight2 * out_image[i][j]
        return result

    def out_address(self,image):
        '''
        마지막 레이어의 주소를 반환
        '''
        out_image = [[0] * self.layer_size for _ in range(self.layer_size)]
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                pattern=[]
                for a in range(self.window_size):
                    for b in range(self.window_size):
                        pattern.append(image[self.stride*i +a][self.stride*j +b])
                out_image[i][j] = max(self.L[i][j].get_address(pattern))
        return out_image

    def get_avg_address(self,image):
        
        out_image = self.out_address(image)
        sum = 0
        for i in range (self.layer_size):
            for j in range(self.layer_size):
                sum = sum + out_image[i][j]

        return sum/(self.layer_size*self.layer_size)
    
    def num_to_pattern(self, num):
        temp = num
        pattern = [0 for _ in range(self.window_size*self.window_size)]
        for i in range(len(pattern)):
            if(num%2) : pattern[i] = 1
            else : pattern[i] = 0
            num = num//2
        pattern.reverse()
        return pattern

    def feature(self, depth):
        out_image = [[0] * self.layer_size *self.window_size for _ in range(self.layer_size*self.window_size)]
        pattern = []
        for i in range(self.layer_size):
            for j in range(self.layer_size):
                for k in range(depth):
                    pattern = self.num_to_pattern(self.L[i][j].cam[-1-k])
                    for a in range(self.window_size):
                        for b in range(self.window_size):
                            out_image[self.window_size*i + a ][self.window_size*j + b ] += pattern[self.window_size*a + b ] * 2**(depth -k-1)
        
        return out_image

    def out_size(self):
        '''
        N*N 출력 이미지에서 N이 나옴 
        '''
        return self.layer_size


# In[10]:


# test_images 를 모두 1차원 벡터화한다. 
test_images_flat = []
temp_1 = [[],[],[],[],[],[]]
temp_2 = [[] for _ in range(80)]

for i in range(10000):
    for j in range(6):
        temp_1[j] = flattening(test_images[i],filters[j])
    temp_2 = edge_pattern(temp_1)
    test_images_flat.append(temp_2)


# In[11]:


import time
from tqdm import tqdm


# In[12]:


Layer_test = [0 for _ in tqdm(range(1000))]
for i in tqdm(range(1000)):
    Layer_test[i] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)

#Layer_test[0].learn(test_images[0])
for i in tqdm(range(1000)):
    for j in range(i):
        if dtw_distance(test_images_flat[j],test_images_flat[i]) < 13 :
            Layer_test[j].learn(test_images[i])
        else :
            Layer_test[i].learn(test_images[i])


# In[ ]:


Layer_test_count = [0 for _ in tqdm(range(1000))]
count = [0 for _ in tqdm(range(1000))]
for i in tqdm(range(1000)):
    Layer_test_count[i] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)

#Layer_test[0].learn(test_images[0])
for i in tqdm(range(1000)):
    for j in range(i):
        if dtw_distance(test_images_flat[j],test_images_flat[i]) < 13 :
            Layer_test_count[j].learn(test_images[i])
            count[j] += 1
        else :
            Layer_test_count[i].learn(test_images[i])
            count[i] += 1

