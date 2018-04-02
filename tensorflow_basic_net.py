
# coding: utf-8

# In[71]:


from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob 
import cv2


class FaceDetect:
    def __init__(self,images,labels):
        #print("Initialized")
        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0
        #print(self.images.shape)
        #print(self.labels.shape)


    #For next batch
    def next_batch(self,batch_size):
        """
        input-batch size
        returns batch_x,batch_y which contain the training and test data of this batch
        batch_x-[batch_size patch_size*patch_size*3]
        batch_y-labels
        index_in_epoch
        """
        #print("Next batch called")
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        #print(self.index_in_epoch)
        if self.index_in_epoch > self.images.shape[0]:
            #print("Shuffling")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.images.shape[0]
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]

        
    
#import data 
patch_size =64
num_training_images=10000
num_testing_images=1000

train_face_data=np.zeros((num_training_images,patch_size*patch_size*3))
train_nonface_data=np.zeros((num_training_images,patch_size*patch_size*3))
labels_face_train=np.ones((1,num_training_images)) #1*50
labels_nonface_train=np.zeros((1,num_training_images)) #1*50

labels_face_test=np.ones((1,num_testing_images)) #1*50
labels_nonface_test=np.zeros((1,num_testing_images)) #1*50

test_face_data=np.zeros((num_testing_images,patch_size*patch_size*3))
test_nonface_data=np.zeros((num_testing_images,patch_size*patch_size*3))

index=0
k=0
for img in glob.glob("Dataset//extracted_faces//*.jpg"):
    if(index==num_testing_images and k==1):
        break
    if(index==num_training_images and k==0):
        print("Training data loaded for face")
        k=1
        index=0
    else:
        n= cv2.imread(img)# Read the face
        if(n.size and k==0):
            train_face_data[index]=n.flatten()
            index=index+1
        elif(n.size and k==1):
            test_face_data[index]=n.flatten()
            index=index+1

print("Training Data Face",train_face_data.shape)
print("Testing Data Face",test_face_data.shape)

index=0
k=0
for img in glob.glob("Dataset//extracted_nonfaces//*.jpg"):
    if(index==num_testing_images and k==1):
        break
    if(index==num_training_images and k==0):
        print("Training data loaded for nonface")
        k=1
        index=0
    else:
        n= cv2.imread(img)# Read the face
        if(n.size and k==0):
            train_nonface_data[index]=n.flatten()
            index=index+1
        elif(n.size and k==1):
            test_nonface_data[index]=n.flatten()
            index=index+1

print("Training Data NonFace",train_nonface_data.shape)
print("Testing Data NonFace",test_nonface_data.shape)

train_all=np.vstack((train_face_data,train_nonface_data))
test_all=np.vstack((test_face_data,test_nonface_data))

labels_l=np.hstack((labels_face_train,labels_nonface_train))
labels_r=np.hstack((labels_nonface_train,labels_face_train))
labels_all=np.vstack((labels_l,labels_r)).T

labels_l=np.hstack((labels_face_test,labels_nonface_test))
labels_r=np.hstack((labels_nonface_test,labels_face_test))
labels_all_test=np.vstack((labels_l,labels_r)).T

#normalizing
train_all=(train_all-np.mean(train_all))/np.var(train_all)
test_all=test_all-np.mean(test_all)/np.var(test_all)
print(train_all.shape)
print(labels_all.shape)
#print(labels_all)


# In[72]:
model = FaceDetect(train_all, labels_all)

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 5


# Network Parameters
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 =  256# 2nd layer number of neurons
n_input = patch_size*patch_size*3 # data input (img shape: 64*64)
n_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, n_input],name="X")
Y = tf.placeholder("float", [None, n_classes],name="Y")
print(X.name)
print(Y.name)
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)
#print(logits.name)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()






# In[ ]:
saver =tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
#        index_in_epoch=0
        total_batch = int(train_all.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #batch_x, batch_y,index_in_epoch,train_all,labels_all = next_batch(batch_size,index_in_epoch,num_training_images,train_all,labels_all)
            batch_x,batch_y=model.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    saver.save(sess, './trained-models/my_final_model')
 #   print("model saved")
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test_all, Y: labels_all_test}))

