#This is to run inference on a saved model with an image
import numpy as np
import cv2
import time
import argparse
import tensorflow as tf
parser = argparse.ArgumentParser(description='Run Inference on image with savedModel.')
parser.add_argument("-m","--model",required=True,type=String,help='path to model')
args=parser.parse_args()
print(args.model)
sess = tf.Session()
saver = tf.train.import_meta_graph('./trained-models/my_final_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./trained-models/'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("add:0")
patch_size = 64
frame = cv2.imread(
    "/home/hariharan/Desktop/Projects/ECE-763-Final-Project/Data/LFW/Zulfiqar_Ahmed/Zulfiqar_Ahmed_0001.jpg")
frame_new = cv2.resize(frame, (patch_size, patch_size), cv2.INTER_LANCZOS4)
frame_new = frame_new.flatten()
frame_new = frame_new[np.newaxis, :]
tic = time.time()
dec = sess.run(Y, feed_dict={X: frame_new})  # decoded output
toc = time.time()
if(np.argmax(dec) == 0):
    decision = "face"
else:
    decision = "non-face"

#print(decision)
cv2.putText(frame, decision, (int(frame.shape[0]/2), int(frame.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
