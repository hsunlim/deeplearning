
# coding: utf-8

# In[1]:

#conv Neural Network
# tensorboard --logdir=/home/ncc/notebook/learn/tensorboard/log
"""
created by kim Seong jung

"""
import numpy as np 
import tensorflow as tf

import math
import time
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os 



file_locate='/home/user01/notebook/Mnist_Data/'


img_row = 28
img_col = 28
in_ch =1


learning_rate=0.0001
batch_size=10
n_classes =10


weight_row =3 ; weight_col=3
out_ch1=100
out_ch2=100
fully_ch1=1024

strides_1=[1,1,1,1]
strides_2=[1,1,1,1]
iterate=50000


device_ = '/gpu:0'



with tf.device(device_):
    x= tf.placeholder("float",shape=[batch_size,img_col * img_row * in_ch],  name = 'x-input')
    y_=tf.placeholder("float",shape=[batch_size, n_classes] , name = 'y-input')
    keep_prob = tf.placeholder("float")
    x_image= tf.reshape(x,[-1,img_row,img_col,in_ch])





print img_col , img_row


# In[2]:

with tf.device('/gpu:2'):
    #with tf.device('/gpu:2'):
    train_img=np.load(file_locate+'train_img.npy');
    train_lab=np.load(file_locate+'train_lab.npy');
    val_img= np.load(file_locate+'val_img.npy');
    val_lab = np.load(file_locate+'val_lab.npy');
    test_img=np.load(file_locate+'test_img.npy');
    test_lab=np.load(file_locate+'test_lab.npy');

    print "Training Data",np.shape(train_img)
    print "Training Data Label",np.shape(train_lab)
    print "Test Data Label",np.shape(test_lab)
    print "val Data Label" , np.shape(val_img)

    n_train= np.shape(train_img)[0]
    n_train_lab = np.shape(train_lab)[0]

def save_parameter(dirname):
    f=open(dirname+"/setting.txt",'w')
    f.write("batch_size:"+str(batch_size)+'\n')
    f.write("img_row:"+str(img_row)+'\n')
    f.write("img_col:"+str(img_col)+'\n')
    f.write("n_classes:"+str(n_classes)+'\n')
    f.write("in_ch:"+str(in_ch)+'\n')
    f.write("out_ch1:"+str(out_ch1)+'\n')
    f.write("out_ch2:"+str(out_ch2)+'\n')
    f.write("fully_ch1:"+str(fully_ch1)+'\n')
    f.write("strides_1:"+str(strides_1)+'\n')
    f.write("strides_2:"+str(strides_2)+'\n')
    f.write("iterate:"+str(iterate)+'\n')
    sf.write("learning_rate:"+str(learning_rate)+'\n')
    
# In[3]:

"""def weight_variable(name,shape):
    #initial = tf.truncated_normal(shape , stddev=0.1)
    initial = tf.get_variable(name,shape=shape , initializer = tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)"""
with tf.device('/gpu:2'):
    def bias_variable(shape):
        initial = tf.constant(0.1 , shape=shape)
        return tf.Variable(initial)


# In[4]:

with tf.device('/gpu:2'):
    def next_batch(batch_size , image , label):

        a=np.random.randint(np.shape(image)[0] -batch_size)
        batch_x = image[a:a+batch_size,:]
        batch_y= label[a:a+batch_size,:]
        return batch_x, batch_y


# In[5]:

with tf.device('/gpu:2'):
    def conv2d(x,w,strides_):
        return tf.nn.conv2d(x,w, strides = strides_, padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x , ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'SAME')


# In[6]:

with tf.device('/gpu:2'):
    with tf.variable_scope("layer1") as scope:
        try:
            w_conv1 = tf.get_variable("W1",[weight_row,weight_col,in_ch,out_ch1] , initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_conv1 = tf.get_variable("W1",[weight_row,weight_col,in_ch,out_ch1] , initializer = tf.contrib.layers.xavier_initializer())
    with tf.variable_scope("layer1") as scope:
        try:
            b_conv1 = bias_variable([out_ch1])
        except:
            scope.reuse_variables()
            b_conv1 = bias_variable([out_ch1])
    with tf.variable_scope('layer2') as scope:
        try:
            w_conv2 = tf.get_variable("W2",[weight_row,weight_col,out_ch1,out_ch2] , initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_conv2 = tf.get_variable("W2",[weight_row,weight_col,out_ch1,out_ch2] , initializer = tf.contrib.layers.xavier_initializer())

    with tf.variable_scope('layer2') as scope:
        try:
            b_conv2= bias_variable([out_ch2])
        except:
            scope.reuse_variables()
            b_conv2= bias_variable([out_ch2])


# In[7]:

#conncect hidden layer 
with tf.device('/gpu:2'):
    h_conv1 = tf.nn.relu(conv2d(x_image , w_conv1 ,strides_1)+b_conv1)
    h_conv1 = max_pool_2x2(h_conv1)#pooling
    h_conv2 = tf.nn.relu(conv2d(h_conv1 , w_conv2 ,strides_2)+b_conv2)
    h_conv2 = max_pool_2x2(h_conv2)#pooling
   
    print h_conv1
    print h_conv2
  

    #print conv2d(h_pool1 , w_conv2).get_shape()


# In[ ]:

with tf.device('/gpu:2'):

    end_conv = h_conv2
    end_conv_row=int(h_conv5.get_shape()[1])
    end_conv_col=int(h_conv5.get_shape()[2])
    end_conv_ch=int(h_conv5.get_shape()[3])


# In[ ]:

#connect fully connected layer 
with tf.device('/gpu:2'):
    with tf.variable_scope("fc1") as scope:
        try:
            w_fc1=tf.get_variable("fc1_W",[end_conv_col*end_conv_row*end_conv_ch,fully_ch1] , initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_fc1=tf.get_variable("fc1_W",[end_conv_col*end_conv_row*end_conv_ch,fully_ch1] , initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_fc1 = bias_variable([fully_ch1])
        except:
            scope.reuse_variables()
            b_fc1 = bias_variable([fully_ch1])

        
with tf.device('/gpu:2'): # flat conv layer 
    end_flat_conv =tf.reshape(end_conv, [-1,end_conv_col*end_conv_row*end_conv_ch])
   
with tf.device('/gpu:2'): # connect flat layer with fully  connnected layer 
    h_fc1 = tf.nn.relu(tf.matmul(end_flat_conv , w_fc1)+ b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)


# In[ ]:

with tf.device('/gpu:2'):
    end_fc=h_fc1
    end_ch=end_fc.get_shape()[1]
    print end_ch


# In[ ]:

with tf.device('/gpu:2'):
    with tf.variable_scope('fc3') as scope:
        try:
            w_end =tf.get_variable("end_W",[end_ch , n_classes ],initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_end =tf.get_variable("end_W",[end_ch , n_classes],initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_end = bias_variable([n_classes])
        except:
            scope.reuse_variables()
            b_end = bias_variable([n_classes])

with tf.device('/gpu:2'):  # join flat layer with fully  connnected layer 
    y_conv = tf.matmul(end_fc , w_end)+b_end
    


# In[ ]:

with tf.device('/gpu:2'):
#sm_conv= tf.nn.softmax(y_conv)
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    start_time = time.time()

    #regular=0.01*(tf.reduce_sum(tf.square(y_conv)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( y_conv, y_))
with tf.device('/gpu:2'):
    #cost = cost+regular
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost) #1e-4
    with tf.name_scope("accuracy"):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv,1) ,tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float")) 

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

def draw_graph_acc(train_acc , val_acc , batch_size,dirname):
    fig , ax =plt.subplots()
    ax.plot(range(0,len(train_acc)*batch_size , batch_size) , train_acc , label='train' )
    ax.plot(range(0,len(val_acc)*batch_size , batch_size) , val_acc , label='val' )
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.6,1)
    ax.set_title('Training Vs Validation Accuracy')
    ax.legend(loc=4)
    np.save(dirname+"/train_acc" ,np.asarray(train_acc))
    np.save(dirname+"/val_acc" ,np.asarray(val_acc))
    plt.savefig(dirname+'/acc_graph')
    plt.show()
    
# In[ ]:

save_parameter(dirname)


# In[ ]:

share = int(len(val_img) / batch_size)
for i in range(iterate):    
    batch_xs , batch_ys = next_batch(batch_size, train_img , train_lab)
   # batch_val_xs  , batch_val_ys = next_batch(20 , val_img , val_lab)
    if i%100 ==0: # in here add to validation 
            for i in range(share):
                val_accuracy ,val_loss= sess.run([accuracy ,cost], feed_dict={x:val_img[i*batch_size : (i+1)*batch_size] ,                                            y_:val_lab[i*batch_size : (i+1)*batch_size] , keep_prob: 1.0})        

            #result = sess.run(sm_conv , feed_dict = {x:val_img , y_:batch_ys , keep_prob :1.0})
            #train_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(train_loss) +'\tval accuracy:\t'+str(train_accuracy)+'\n' 
            print("step %d , validation  accuracy %g" %(i,val_accuracy))
            print("step %d , validation loss : %g" %(i,val_loss))
            #val_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(val_loss) +'\tval accuracy:\t'+str(val_accuracy)+'\n'
            
            
            #f.write(val_str)
            #f.write(train_str)
    sess.run(train_step ,feed_dict={x:batch_xs , y_:batch_ys , keep_prob : 0.7})
print("--- Training Time : %s ---" % (time.time() - start_time))
train_time="--- Training Time : ---:\t" +str(time.time() - start_time)
f.write(train_time)


# In[ ]:

draw_graph_acc(list_train_acc,list_val_acc,100,dirname)


# In[ ]:

try:
    test_accuracy = sess.run( accuracy , feed_dict={x:test_img , y_:test_lab , keep_prob: 1.0})        
    test_loss = sess.run(cost , feed_dict = {x:test_img , y_: test_lab , keep_prob: 1.0})

    #result = sess.run(sm_conv , feed_dict = {x:test_img , y_:batch_ys , keep_prob :1.0})
    print("step %d , testidation  accuracy %g" %(i,test_accuracy))
    print("step %d , testidation loss : %g" %(i,test_loss))
    test_str = 'step:\t'+str(i)+'\ttest_loss:\t'+str(test_loss) +'\ttest accuracy:\t'+str(test_accuracy)+'\n'

    f.write(test_str)
except :
    list_acc=[]
    list_loss=[]
    n_divide=len(test_img)/batch_size
    for j in range(n_divide):

        # j*batch_size :(j+1)*batch_size
        test_accuracy,test_loss = sess.run([accuracy ,cost], feed_dict={x:test_img[ j*batch_size :(j+1)*batch_size] , y_:test_lab[ j*batch_size :(j+1)*batch_size ] , keep_prob: 1.0})        
        list_acc.append(float(test_accuracy))
        list_loss.append(float(test_loss))
    test_accuracy , test_loss=sess.run([accuracy,cost] , feed_dict={x:test_img[(j+1)*batch_size : ] , y_:test_lab[(j+1)*(batch_size) : ] , keep_prob : 1.0})
    #right above code have to modify

    list_acc.append(test_accuracy)
    list_loss.append(test_loss)
    list_acc=np.asarray(list_acc)
    list_loss= np.asarray(list_loss)

    test_accuracy=np.mean(list_acc)
    test_loss = np.mean(list_loss)

    #result = sess.run(sm_conv , feed_dict = {x:test_img , y_:batch_ys , keep_prob :1.0})
    print("step %d , testidation  accuracy %g" %(i,test_accuracy))
    print("step %d , testidation loss : %g" %(i,test_loss))
    test_str = 'step:\t'+str(i)+'\ttest_loss:\t'+str(test_loss) +'\ttest accuracy:\t'+str(test_accuracy)+'\n'

    f.write(test_str)


# In[ ]:

sess.close()

