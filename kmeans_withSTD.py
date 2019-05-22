from facenet.src import facenet
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd
import cv2
import os
with tf.Graph().as_default():
  with tf.Session() as sess:
    np.random.seed(seed=666)
    os.system("python facenet/src/align/align_dataset_mtcnn.py "+sys.argv[1]+" aligned --image_size 160 --margin 32 --random_order")
    dataset = facenet.get_dataset("aligned")
    paths, labels = facenet.get_image_paths_and_labels(dataset)
    facenet.load_model('20180402-114759.pb')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    images = facenet.load_data(paths, False, False, 150)
    emb_array_1 = np.zeros((len(labels), embedding_size))
    j=min(len(images)-1,500)
    i=0
    while j<len(images):
        feed_dict = { images_placeholder:images[i:j], phase_train_placeholder:False }
        emb_array_1[i:j,:] = sess.run(embeddings, feed_dict=feed_dict)
        i=j
        j+=500
    model = KMeans(n_clusters=max(labels)+1, random_state=40, max_iter=1000)
    model.fit(emb_array_1, labels)
    class_names = [ cls.name.replace('_', ' ') for cls in dataset]
    classes={x:[] for x in class_names}
    for i, clas in enumerate(model.labels_):
      classes[class_names[clas]].append(i)
    print(model.cluster_centers_)
    vidcap = cv2.VideoCapture(sys.argv[2])
    nrof_images=0

    success,image = vidcap.read()
    images_1=[]
    while success:
    #   print(image.shape)
      images_1.append(image)
      cv2.imwrite("imgs/imgs/penny"+str(nrof_images)+".jpg", image)
      success,image = vidcap.read()
      if not success:
        break
      nrof_images += 1
    os.system("python facenet/src/align/align_dataset_mtcnn.py imgs images --image_size 160 --margin 32 --random_order")

    dataset=facenet.get_dataset('images')
    print(len(dataset), nrof_images)

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    images=facenet.load_data(paths, False, False, 160)
    images_placeholder_1 = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings_1 = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder_1 = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    print('Calculating features for images')
    nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / 200))
    emb_array = np.zeros((len(images), embedding_size))
    feed_dict_1 = { images_placeholder_1:images, phase_train_placeholder_1:False }
    emb_array[:,:] = sess.run(embeddings_1, feed_dict=feed_dict_1)
    for j,img_arr in enumerate(emb_array):
        stds={}
        for i,center in enumerate(model.cluster_centers_):
            std=np.std([img_arr]+[center], axis=0)
            stds[class_names[i]]=std
        df=pd.DataFrame.from_dict(stds)
        df.to_excel("output/frame"+str(j)+".xlsx",index=True)
