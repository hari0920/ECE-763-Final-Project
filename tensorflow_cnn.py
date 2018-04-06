import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import glob
import cv2


class FaceDetect:
    def __init__(self, images, labels):
        #print("Initialized")
        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0
        #print(self.images.shape)
        #print(self.labels.shape)

    #For next batch
    def next_batch(self, batch_size):
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
patch_size = 64
num_training_images = 10000
num_testing_images = 1000

train_face_data = np.zeros((num_training_images, patch_size*patch_size*3))
train_nonface_data = np.zeros((num_training_images, patch_size*patch_size*3))
labels_face_train = np.ones((1, num_training_images))  # 1*50
labels_nonface_train = np.zeros((1, num_training_images))  # 1*50

labels_face_test = np.ones((1, num_testing_images))  # 1*50
labels_nonface_test = np.zeros((1, num_testing_images))  # 1*50

test_face_data = np.zeros((num_testing_images, patch_size*patch_size*3))
test_nonface_data = np.zeros((num_testing_images, patch_size*patch_size*3))

index = 0
k = 0
for img in glob.glob("Dataset//extracted_faces//*.jpg"):
    if(index == num_testing_images and k == 1):
        break
    if(index == num_training_images and k == 0):
        print("Training data loaded for face")
        k = 1
        index = 0
    else:
        n = cv2.imread(img,-1)  # Read the face
        if(n.size and k == 0):
            train_face_data[index] = n.flatten()
            index = index+1
        elif(n.size and k == 1):
            test_face_data[index] = n.flatten()
            index = index+1

print("Training Data Face", train_face_data.shape)
print("Testing Data Face", test_face_data.shape)

index = 0
k = 0
for img in glob.glob("Dataset//extracted_nonfaces//*.jpg"):
    if(index == num_testing_images and k == 1):
        break
    if(index == num_training_images and k == 0):
        print("Training data loaded for nonface")
        k = 1
        index = 0
    else:
        n = cv2.imread(img,-1)  # Read the face
        if(n.size and k == 0):
            train_nonface_data[index] = n.flatten()
            index = index+1
        elif(n.size and k == 1):
            test_nonface_data[index] = n.flatten()
            index = index+1

print("Training Data NonFace", train_nonface_data.shape)
print("Testing Data NonFace", test_nonface_data.shape)

train_all = np.vstack((train_face_data, train_nonface_data))
test_all = np.vstack((test_face_data, test_nonface_data))

labels_l = np.hstack((labels_face_train, labels_nonface_train))
labels_r = np.hstack((labels_nonface_train, labels_face_train))
labels_all = np.vstack((labels_l, labels_r)).T

labels_l = np.hstack((labels_face_test, labels_nonface_test))
labels_r = np.hstack((labels_nonface_test, labels_face_test))
labels_all_test = np.vstack((labels_l, labels_r)).T

#normalizing
train_all = (train_all-np.mean(train_all))/np.std(train_all)
test_all = (test_all-np.mean(test_all))/np.std(test_all)
print(train_all.shape)
print(labels_all.shape)
#print(labels_all)

#shuffling
perm = np.arange(train_all.shape[0])
np.random.shuffle(perm)
train_all = train_all[perm]
labels_all = labels_all[perm]

model = FaceDetect(train_all, labels_all)

# Training Parameters
learning_rate = 0.001
training_epochs = 50000
batch_size = 128
display_step = 10
threshold=0.01
logs_path='./logs/cnn/'

# Network Parameters
num_input = patch_size*patch_size*3 # data input (img shape: patch_size*patch_size)
num_classes = 2  # total classes
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input]) #batch_size, num_input
Y = tf.placeholder(tf.float32, [None, num_classes]) #batch_size, num_classes
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# create Lenet model
def Lenet(x,weights,biases,dropout):
    #reshaping
    x=tf.reshape(x,shape=[-1,patch_size,patch_size,3])
    #convolution layer 1
    conv1=conv2d(x,weights['wc1'],biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #fc1=flatten(conv2)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2= tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    fc2=tf.nn.relu(fc2)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out

    """
    Input
    x->contains the images in (patch_size,patch_size,3) format
    The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.
    Modifying to accept 64x64 images 
    
    Architecture
    Layer 1: Convolutional. The output shape should be 28x28x6.

    Activation. Your choice of activation function.

    Pooling. The output shape should be 14x14x6.

    Layer 2: Convolutional. The output shape should be 10x10x16.

    Activation. Your choice of activation function.

    Pooling. The output shape should be 5x5x16.

    Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

    Layer 3: Fully Connected. This should have 120 outputs.

    Activation. Your choice of activation function.

    Layer 4: Fully Connected. This should have 84 outputs.

    Activation. Your choice of activation function.

    Layer 5: Fully Connected (Logits). This should have 10 outputs.
    """


#Lenet
# Store layers weight & bias
lenet_weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 6])),
    # 5x5 conv, 6 inputs, 16 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16])),
    # fully connected, 16*16*16 inputs, 400 outputs
    'wd1': tf.Variable(tf.random_normal([(16)*(16)*16, 400])),
    # fully connected 400 inputs, 120 outputs 
    'wd2': tf.Variable(tf.random_normal([400,120])),
    #fully connected 120 inputs, 84 outputs
    'wd3': tf.Variable(tf.random_normal([120,84])),
    #fully connected 84 inputs, 2 outputs
    'out': tf.Variable(tf.random_normal([84,num_classes]))
}

lenet_biases = {
    'bc1': tf.Variable(tf.random_normal([6])),
    'bc2': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([400])),
    'bd2': tf.Variable(tf.random_normal([120])),
    'bd3': tf.Variable(tf.random_normal([84])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, patch_size, patch_size, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



# Store layers weight & bias
weights = {
    # 4x4 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([4, 4, 3, 32])),
    # 4x4 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
    # fully connected, 16*16*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(16)*(16)*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

with tf.name_scope('model'):
    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    #logits = Lenet(X, lenet_weights, lenet_biases, keep_prob)
    prediction = tf.nn.softmax(logits)
with tf.name_scope('loss'):
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    
    train_op = optimizer.minimize(loss_op)

with tf.name_scope('Accuracy'):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tf.summary.scalar("loss",loss_op)

tf.summary.scalar("accuracy",accuracy)
merged_summary_op=tf.summary.merge_all()

saver =tf.train.Saver()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    for step in range(1, training_epochs+1):
        avg_cost=0
        total_batch = int(train_all.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = model.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x,
                                        Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                    Y: batch_y,
                                                                keep_prob: 1.0})
            # Compute average loss
            #avg_cost += loss / total_batch
            print("Step " + str(step) + ", Minibatch Loss= " +
                "{:.4f}".format(loss) + ", Training Accuracy= " +
                "{:.3f}".format(acc))
            

    print("Optimization Finished!")
    saver.save(sess, './trained-models/cnn/my_Lenet_model')
    # Calculate accuracy for test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_all,
                                        Y: labels_all_test,
                                        keep_prob: 1.0}))
