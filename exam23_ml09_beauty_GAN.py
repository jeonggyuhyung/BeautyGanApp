#!/usr/bin/env python
# coding: utf-8

# In[7]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # 옛날 버전이라 이걸 해주어야 함
import numpy as np


# In[8]:


detector = dlib.get_frontal_face_detector() # 얼굴 찾아주는 것
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat') # 머신러닝으로 만들어진 모델


# In[31]:


# 이미지 로드
img = dlib.load_rgb_image('../imgs/12.jpg')

# 출력
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


# In[32]:


img_result = img.copy()

# 얼굴 찾기
dets = detector(img, 1)
if len(dets) == 0: 
    print('cannot find faces!')
else: 
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:  # det에는 얼굴의 영역들이 들어있음.
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        # rect를 plot에 띄움
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


# In[33]:


fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()


# In[34]:


# 얼굴만 따로 뽑아오기(가능한 일직선 상(수평에 맞게끔))
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[46]:


# 전체코드

def align_faces(img):
    dets = detector(img, 1) # detector는 얼굴 찾아주는 변수 //dets는 얼굴들이 들어있음
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35) # get_face_chips는 얼굴 영역만 잘라서 이미지로 만들어 준다.
    return faces
test_img = dlib.load_rgb_image('../imgs/02.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


# In[50]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 임포트
saver = tf.train.import_meta_graph('../models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[51]:


# 스케일링
def preprocess(img):
    return (img / 255. - 0.5) * 2

# 다시 원본 이미지로 만듬
def deprocess(img):
    return (img + 1) / 2


# In[67]:


# 소스 이미지
img1 = dlib.load_rgb_image('../imgs/12.jpg')
img1_faces = align_faces(img1)

# 레퍼런스 이미지
img2 = dlib.load_rgb_image('../imgs/makeup/vFG56.png')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[68]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0) # shape을 똑같이 맞춰줌

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()

