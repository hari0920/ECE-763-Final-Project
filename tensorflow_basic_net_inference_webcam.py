#This is to run inference on a saved model
import numpy as np
import cv2
import time
import tensorflow as tf
sess=tf.Session()
saver = tf.train.import_meta_graph('./trained-models/my_final_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./trained-models/'))
graph = tf.get_default_graph()
X=graph.get_tensor_by_name("X:0")
Y=graph.get_tensor_by_name("add:0")
cap = cv2.VideoCapture(0)
patch_size=64
while(True):
    ret,frame=cap.read()
    frame_new=cv2.resize(frame, (patch_size, patch_size), cv2.INTER_LANCZOS4)
    frame_new=frame_new.flatten()
    frame_new=frame_new[np.newaxis,:]
    tic=time.time()
    dec = sess.run(Y, feed_dict={X: frame_new}) # decoded output
    #print(sess.run(tf.nn.softmax(dec)))
    toc=time.time()
    if(np.argmax(dec)==0):
        decision="face"
    else:
        decision="non-face"
    cv2.putText(frame, decision, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
    cv2.imshow("Frame", frame)
    if(cv2.waitKey(30)==ord('q')):
        break


cap.release()
cv2.destroyAllWindows()