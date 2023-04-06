#!/usr/bin/env python
# coding: utf-8

# # 이미지 전처리
# 

# In[1]:


from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
#import cv2


# ## 이미지 download / resize

# In[2]:



(train_images, train_labels),(test_images, test_labels)= keras.datasets.mnist.load_data() 

image_size = 28
train_images = np.expand_dims(train_images, axis = -1)
train_images = tf.image.resize(train_images,[image_size,image_size])
train_images = np.squeeze(train_images)
train_images= train_images/255.0

test_images = np.expand_dims(test_images, axis = -1)
test_images = tf.image.resize(test_images,[image_size,image_size])
test_images = np.squeeze(test_images)
test_images= test_images/255.0
#이미지 resize




# ## 이미지 양자화(수정됨)(test를 위해 dataset 줄임)
# 0.3, 0.7 기준으로 (0 / 0.5 / 1)로 나누기

# In[3]:


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
# 0 0.5 1 셋 중 하나의 값으로 바꿔줌


# ## 데이터 나누기

# In[4]:


###################데이터 나누기 
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


# # Random Chip / Layer 선언

# ## Random Chip

# In[5]:


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



# ## Random Layer

# In[6]:


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


# #Edge Detection

# ## Filtering CAM(`Chip`) 학습(수정됨)
# size 4의 Chip을 선언하고 수평, 수직, 반대각, 주대각을 학습하는 데이터를 학습한다.
# 

# In[7]:


edge_detection = chip_random(4,threshold1 = 14/16, threshold2=14/16)

#수평 = H
for i in range (8):
    edge_detection.learn([1,1,0,0])
    edge_detection.learn([0,0,1,1])
#수직 = V
for i in range (6):
    edge_detection.learn([1,0,1,0])
    edge_detection.learn([0,1,0,1])
#반대각(ㅢ) = R
for i in range (4):
    edge_detection.learn([1,0,0,0])
    edge_detection.learn([0,1,1,1])
#주대각(ㄴ) = L
for i in range (2):
    edge_detection.learn([0,1,0,0])
    edge_detection.learn([1,0,1,1])


# ## Edge Detection : `Chip` 배열
# H, V, R, L의 배열(0 - 9)을 선언하고 각 배열의 element를 27*27로 만들어 준다.
# 
# 전처리한 image(`data`)를 입력해 수평/수직/반대각/주대각(H, V, R, L)의 배열에 해당 패턴이 발견되면 1, 발견되지 않으면 0으로 입력한다.
# 
# `k`는 0 - 9
# 
# `a`는 각 숫자의 mnist 이미지
# 
# `i`, `j`는 이미지의 위치
# 

# In[8]:


H = [[],[],[],[],[],[],[],[],[],[]]
V = [[],[],[],[],[],[],[],[],[],[]]
R = [[],[],[],[],[],[],[],[],[],[]]
L = [[],[],[],[],[],[],[],[],[],[]]


for k in range(10):
    for a in range(len(train[k])) :
        data = train[k][a]    
        H[k].append([[0] * 27 for _ in range(27)])
        V[k].append([[0] * 27 for _ in range(27)])
        R[k].append([[0] * 27 for _ in range(27)])
        L[k].append([[0] * 27 for _ in range(27)])
        for i in range(27):
            for j in range(27):
                address = []
                pattern = []
                pattern = [(data[i][j]),(data[i][j+1]),
                          (data[i+1][j]),(data[i+1][j+1])]
                address = edge_detection.get_address(pattern)
                if 15 in address:
                    H[k][a][i][j] = 1
                elif 14 in address:
                    H[k][a][i][j] = 1
                else : H[k][a][i][j] = 0

                if 13 in address:
                    V[k][a][i][j] = 1
                elif 12 in address:
                    V[k][a][i][j] = 1
                else : V[k][a][i][j] = 0

                if 11 in address:
                    R[k][a][i][j] = 1
                elif 10 in address:
                    R[k][a][i][j] = 1
                else : R[k][a][i][j] = 0

                if 9 in address:
                    L[k][a][i][j] = 1
                elif 8 in address:
                    L[k][a][i][j] = 1
                else : L[k][a][i][j] = 0
                


# # Memory Learning(수정됨)

# ## 3x3 Pattern CAM(Layer) 학습
# 
# Edge detection을 완료한 image(`data in H[k] - L[k]`)를 Layer_H1[0] - Layer_H1[9], Layer_V1[0 - 9], Layer_R1[0 - 9], Layer_L1[0 - 9]에 대해 학습(`learn`)
# 
# `window size`=3
# `layer size`=9

# ### 수정 전 코드

# Layer_H1 = [0,0,0,0,0,0,0,0,0,0]
# for k in range(10):
#     Layer_H1[k] = layer(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     for data in H[k]:
#         Layer_H1[k].learn(data)
#         
# Layer_V1 = [0,0,0,0,0,0,0,0,0,0]
# for k in range(10):
#     Layer_V1[k] = layer(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     for data in V[k]:
#         Layer_V1[k].learn(data)
# 
# Layer_R1 = [0,0,0,0,0,0,0,0,0,0]
# for k in range(10):
#     Layer_R1[k] = layer(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     for data in R[k]:
#         Layer_R1[k].learn(data)
# 
# Layer_L1 = [0,0,0,0,0,0,0,0,0,0]
# for k in range(10):
#     Layer_L1[k] = layer(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     for data in L[k]:
#         Layer_L1[k].learn(data)

# ### 수정 후 코드

# In[15]:


'''
learning group이 어떤 이미지에 잘 반응하는지 test
'''
#from tkinter import N




'''
10개의 3x3 Pattern CAM Layer 생성
'''
Layer_H1 = [0,0,0,0,0,0,0,0,0,0]
Layer_V1 = [0,0,0,0,0,0,0,0,0,0]
Layer_R1 = [0,0,0,0,0,0,0,0,0,0]
Layer_L1 = [0,0,0,0,0,0,0,0,0,0]


for k in range(10):
    Layer_H1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
    Layer_V1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
    Layer_R1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
    Layer_L1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)

'''
'0 ~ 9' edge detection image를 10개의 layer에 입력
'''
for N in range(300):    # for i in range(10):

  save_out_result = []
  save_out_result_T = [[],[],[],[],[],[],[],[],[],[]]
  rank = []

  for M in range(10):    # for j in range(len(H[i])):
    data_H = H[M][N]
    data_V = V[M][N]
    data_R = R[M][N]
    data_L = L[M][N]

    #Layer_H1[k].learn(data)
    '''
    합산하여 각 layer의 sum out 저장
    '''
    out_result=[]

    for k in range(10):
      out_H = Layer_H1[k].out(data_H)
      out_V = Layer_V1[k].out(data_V)
      out_R = Layer_R1[k].out(data_R)
      out_L = Layer_L1[k].out(data_L)

      out_H_ = sum(out_H,[])
      out_V_ = sum(out_V,[])
      out_R_ = sum(out_R,[])
      out_L_ = sum(out_L,[])

      out = sum(out_H_)+sum(out_V_)+sum(out_R_)+sum(out_L_)
      out_result.append(out)
    
    #answer = out_result.index(max(out_result))

    save_out_result.append(out_result)



  for i in range(10):
    for j in range(10):
      save_out_result_T[i].append(save_out_result[j][i])
      
  for k in range(10):
    print(save_out_result[k])
  print()
  for k in range(10):
    print(save_out_result_T[k])
  print()
  for k in range(10):
    rank.append(save_out_result_T[k].index(max(save_out_result_T[k])))
  print(rank)

  '''
  각 layer를 sum out이 가장 컸던 이미지로 학습
  '''
  for k in range(10) :
    #for i in range(1) :
    Layer_H1[k].learn(H[rank[k]][N])
    Layer_V1[k].learn(V[rank[k]][N])
    Layer_R1[k].learn(R[rank[k]][N])
    Layer_L1[k].learn(L[rank[k]][N])




# In[ ]:



#save_out_result = []

'''
'0 ~ 9' edge detection image를 10개의 layer에 입력
'''
for i in range(10):    # for i in range(10):
  for j in range(600):    # for j in range(len(H[i])):
    data_H = H[i][j]
    data_V = V[i][j]
    data_R = R[i][j]
    data_L = L[i][j]

    #Layer_H1[k].learn(data)
    '''
    합산하여 가장 높은 layer 찾기
    '''
    out_result=[]

    for k in range(10):
      out_H = Layer_H1[k].out(data_H)
      out_V = Layer_V1[k].out(data_V)
      out_R = Layer_R1[k].out(data_R)
      out_L = Layer_L1[k].out(data_L)

      out_H_ = sum(out_H,[])
      out_V_ = sum(out_V,[])
      out_R_ = sum(out_R,[])
      out_L_ = sum(out_L,[])

      out = sum(out_H_)+sum(out_V_)+sum(out_R_)+sum(out_L_)
      out_result.append(out)
    
    answer = out_result.index(max(out_result))

    '''
    #가장 높은 layer를 학습(최상위 주소로 up)
    ''' 
    Layer_H1[answer].learn(data_H)            
    Layer_V1[answer].learn(data_V)
    Layer_R1[answer].learn(data_R)
    Layer_L1[answer].learn(data_L)

    #save_out_result.append(out_result)

'''
    print("out_result : ", end='')
    print(out_result)

    print("%d rank : " % (i), end='')
    out_rank = ss.rankdata(out_result,'max')
    for l in range(10) :
      if (out_rank[l] < 10) :
        out_rank[l] = 0
      else :
        print("%d " %(l), end='')
    print()
'''
    
'''
    if(j == 0) :
      print(answer)
      before_learning = np.array(Layer_H1[answer].L[0][0].cam)
      #print(type(before_learning))
      print(data_H)
      plt.imshow(data_H)
      print(before_learning)
      print(len(before_learning))
      
    #print(answer)

'''
    #가장 높은 layer를 학습(최상위 주소로 up)
''' 
    Layer_H1[answer].learn(data_H)            ##-> 수정
    Layer_V1[answer].learn(data_V)
    Layer_R1[answer].learn(data_R)
    Layer_L1[answer].learn(data_L)

    if(j == 0) :
      after_learning = np.array(Layer_H1[answer].L[0][0].cam)
      print(after_learning)
      print(before_learning - after_learning)
      
''' 


# Layer_H1 = [0,0,0,0,0,0,0,0,0,0]
# Layer_V1 = [0,0,0,0,0,0,0,0,0,0]
# Layer_R1 = [0,0,0,0,0,0,0,0,0,0]
# Layer_L1 = [0,0,0,0,0,0,0,0,0,0]
# 
# 
# for k in range(10):
#     Layer_H1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     Layer_V1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     Layer_R1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
#     Layer_L1[k] = layer_random(3,9, stride= 0, threshold1 = 0.90, threshold2 = 0.96)
# '''
# 0 ~ 9 edge detection image 입력 -> for문 수정 필요
# '''
# for j in range(10):
#   for data_H in H[j]:
#     for data_V in V[j]:
#       for data_R in R[j]:
#         for data_L in L[j]:
#           #Layer_H1[k].learn(data)
#           '''
#           합산하여 가장 높은 layer 찾기
#           '''
#           out_result=[]
# 
#           for k in range(10):
#             out_H = Layer_H1[k].out(data_H)
#             out_V = Layer_V1[k].out(data_V)
#             out_R = Layer_R1[k].out(data_R)
#             out_L = Layer_L1[k].out(data_L)
# 
#             out_H_ = sum(out_H,[])
#             out_V_ = sum(out_V,[])
#             out_R_ = sum(out_R,[])
#             out_L_ = sum(out_L,[])
# 
#             out = sum(out_H_)+sum(out_V_)+sum(out_R_)+sum(out_L_)
#             out_result.append(out)
#           
#           answer = out_result.index(max(out_result))
#           '''
#           가장 높은 layer를 학습
#           '''
#           Layer_H1[answer].learn(data_H)
#           Layer_V1[answer].learn(data_V)
#           Layer_R1[answer].learn(data_R)
#           Layer_L1[answer].learn(data_L)
#         
# 

# # Model Validation

# ##Validation data Edge detection / Searching
# 위의 H, V, R, L과 동일한 방식으로 H_temp, V_temp, R_temp, L_temp 학습
# 
# i = 0부터 9까지 10번 반복, Layer_H1[0] - Layer_H1[9], Layer_V1[0 - 9], Layer_R1[0 - 9], Layer_L1[0 - 9]에 대해 반복 수행
# 
# H,V,R,L 중 가장 큰 값을 도출하는(maximum of (`H[0] + V[0] + R[0] + L[0]` - `H[9] + V[9] + R[9] + L[9]`)) 값을 `answer`로 출력.
# 
# 정답과 같으면 `right = right + 1`, 틀리면 `wrong = wrong + 1`
# 

# In[ ]:


right = 0
wrong = 0
H_temp = [[0] * 27 for _ in range(27)]
V_temp = [[0] * 27 for _ in range(27)]
R_temp = [[0] * 27 for _ in range(27)]
L_temp = [[0] * 27 for _ in range(27)]
confusion = [[0] * 10 for _ in range(10)]

for k in range(len(test_images)):
    data = test_images[k]
    out_address = []
    out_result=[]

    for i in range(27):
        for j in range(27):
            pattern = []
            pattern = [(data[i][j]),(data[i][j+1]),
                      (data[i+1][j]),(data[i+1][j+1])]
            address = edge_detection.get_address(pattern)
            if 15 in address:
                H_temp[i][j] = 1
            elif 14 in address:
                H_temp[i][j] = 1
            else : H_temp[i][j] = 0

            if 13 in address:
                V_temp[i][j] = 1
            elif 12 in address:
                V_temp[i][j] = 1
            else : V_temp[i][j] = 0

            if 11 in address:
                R_temp[i][j] = 1
            elif 10 in address:
                R_temp[i][j] = 1
            else : R_temp[i][j] = 0
            
            if 9 in address:
                L_temp[i][j] = 1
            elif 8 in address:
                L_temp[i][j] = 1
            else : L_temp[i][j] = 0

#여기까지 H_temp, V_temp, R_temp, L_temp 초기화
#여기부터 i = 0부터 9까지 10번 반복, Layer_H1[0]부터 Layer_H1[9]까지, V, R, L까지 반복 수행

    for i in range(10):
        out_H = Layer_H1[i].out(H_temp)
        out_V = Layer_V1[i].out(V_temp)
        out_R = Layer_R1[i].out(R_temp)
        out_L = Layer_L1[i].out(L_temp)

        out_H_ = sum(out_H,[])
        out_V_ = sum(out_V,[])
        out_R_ = sum(out_R,[])
        out_L_ = sum(out_L,[])

        out = sum(out_H_)+sum(out_V_)+sum(out_R_)+sum(out_L_)
        out_result.append(out)
        
    answer = out_result.index(max(out_result))

    #confusion[test_labels[k]][answer] = confusion[answer][test_labels[k]] + 1
    confusion[answer][test_labels[k]] = confusion[answer][test_labels[k]] + 1
    if answer == test_labels[k]:
        right = right +1
    else :
        wrong = wrong + 1

print(right / (right + wrong))


# In[ ]:


for i in range(10):
    print(confusion[i])


# ## 결과 시각화(table)

# In[ ]:


'''
import pandas as pd
conf = [0,0,0,0,0,0,0,0,0,0]

for i in range(10):
    conf[i] = 100/sum(confusion[i])

for i in range(10):
    for j in range(10):
        confusion[i][j] = confusion[i][j]*conf[i]
    
df = pd.DataFrame(data = {'label_0' : confusion[0],
                          'label_1' : confusion[1],
                          'label_2' : confusion[2],
                          'label_3' : confusion[3], 
                          'label_4' : confusion[4],
                          'label_5' : confusion[5],
                          'label_6' : confusion[6],
                          'label_7' : confusion[7],
                          'label_8' : confusion[8],
                          'label_9' : confusion[9]
                         }
                 ,index = ['0','1','2','3','4','5','6','7','8','9']
                 )

df
'''


# In[ ]:


#confusion.size()


# In[ ]:




