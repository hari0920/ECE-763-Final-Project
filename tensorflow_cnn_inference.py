#This is to run inference on a saved model with an image
import numpy as np
import cv2
import time
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Inference on image with saved Model')
parser.add_argument('-i','--image',help='Path to image',required=True)
parser.add_argument('-m','--model',help='Path to model',required=True)
parser.add_argument('-c','--checkpoint',help='Path to checkpoint',required=True)
args = vars(parser.parse_args())
image_path=args['image']
model_path=args['model']
checkpoint_path=args['checkpoint']
sess = tf.Session()
saver = tf.train.import_meta_graph(model_path)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("model/Softmax:0")
keep_prob=graph.get_tensor_by_name("dropout:0")
patch_size = 64
frame = cv2.imread(image_path)
frame_new = cv2.resize(frame, (patch_size, patch_size), cv2.INTER_LANCZOS4)
frame_new = frame_new.flatten()
frame_new = frame_new[np.newaxis, :]
tic = time.time()
dec = sess.run(Y, feed_dict={X: frame_new,keep_prob:1.0})  # decoded output
toc = time.time()
if(np.argmax(dec) == 0):
    decision = "face"
else:
    decision = "non-face"

print(decision)
cv2.putText(frame, decision, (int(frame.shape[0]/2), int(frame.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 7, 255)
cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
