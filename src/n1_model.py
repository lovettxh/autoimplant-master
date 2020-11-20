from glob import glob
from conv3 import *
import numpy as np
import nrrd
import os
import random

# load data for training and testing.
from data_loader import *
# pre_post_processing use 3D connected component analysis to remove
# the isolated noise from the predictions. 
from pre_post_processing import *


#**************************************************************

# the codes are adapted from 
# https://link.springer.com/chapter/10.1007/978-3-319-75541-0_23
# the network architecture/data loader/loss function is adapted

#**************************************************************



class auto_encoder(object):
    def __init__(self, sess):
        self.sess           = sess
        self.phase          = 'train'
        self.batch_size     = 1
        self.inputI_size    = 128
        self.inputI_chn     = 1
        self.output_chn     = 2
        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 10000
        self.model_name     = 'n1.model'
        self.save_intval    = 100
        
        self.build_model()
        self.init_train_set()
        

        # directory where the checkpoint can be saved/loaded
        self.chkpoint_dir   = r"D:\autoimplant-master\ckpt"
        # directory containing the 100 training defective skulls
        self.train_data_dir = r"D:\autoimplant-master\training_set\defective_skull"
        # ground truth (implants) for the training data
        self.train_label_dir = r"D:\autoimplant-master\training_set\implant"
        # test data directory
        self.test_data_dir = r"D:\autoimplant-master\training_set\defective_skull"
        #self.test_data_dir = r"D:\autoimplant-master\test_set_for_participants"
        # where to save the predicted implants
        self.save_dir = r"D:\autoimplant-master\predictions_n1\\"


     # 3D dice loss function 
     # credits to (https://link.springer.com/chapter/10.1007/978-3-319-75541-0_23)
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 2)
        dice = 0
        for i in range(2):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice


    def build_model(self):
        print('building patch based model...')       
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.inputI_size,self.inputI_size,64, self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,self.inputI_size,self.inputI_size,64,1], name='target')
        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        self.saver = tf.train.Saver()

    #-------------------
    def init_train_set(self):
        r = {i for i in range(100)}
        #self.valid_set = {random.randint(0, 99) for _ in range(10)}
        self.valid_set = set()
        while(len(self.valid_set) < 10):
            rand = random.randint(0, 99)
            if(not (rand in self.valid_set)):
                self.valid_set.add(rand)
        self.train_set = r - self.valid_set
    #-------------------
    
    # def encoder_decoder(self, inputI):
    #     phase_flag = (self.phase=='train')
    #     #inputI (1, 256, 256, 128,1)
    #     print('0',inputI.shape)
    #     conv1_1 = conv3d(input=inputI, output_chn=64, kernel_size=5, stride=2, use_bias=True, name='conv1')
    #     conv1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv1_batch_norm")
    #     conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
    #     print('1',conv1_relu.shape)
    #     conv2_1 = conv3d(input=conv1_relu, output_chn=128, kernel_size=5, stride=2, use_bias=True, name='conv2')
    #     conv2_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv2_batch_norm")
    #     conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')
    #     print('2',conv2_relu.shape)
    #     conv3_1 = conv3d(input=conv2_relu, output_chn= 256, kernel_size=5, stride=2, use_bias=True, name='conv3a')
    #     conv3_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_1_batch_norm")
    #     conv3_relu = tf.nn.relu(conv3_bn, name='conv3_1_relu')
    #     print('3',conv3_relu.shape)
    #     conv4_1 = conv3d(input=conv3_relu, output_chn=512, kernel_size=5, stride=2, use_bias=True, name='conv4a')
    #     conv4_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_1_batch_norm")
    #     conv4_relu = tf.nn.relu(conv4_bn, name='conv4_1_relu')
    #     print('4',conv4_relu.shape)
    #     conv5_1 = conv3d(input=conv4_relu, output_chn=512, kernel_size=5, stride=1, use_bias=True, name='conv5a')
    #     conv5_bn = tf.contrib.layers.batch_norm(conv5_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv5_1_batch_norm")
    #     conv5_relu = tf.nn.relu(conv5_bn, name='conv5_1_relu')
    #     print('5',conv5_relu.shape)
    #     feature= conv_bn_relu(input=conv5_relu, output_chn=256, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='conv6_1')
    #     print('feature',feature.shape)
    #     deconv1_1 = deconv_bn_relu(input=feature, output_chn=256, is_training=phase_flag, name='deconv1_1')
    #     deconv1_2 = conv_bn_relu(input=deconv1_1, output_chn=128, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_2')
    #     print('6',deconv1_2.shape)
    #     deconv2_1 = deconv_bn_relu(input=deconv1_2, output_chn=128, is_training=phase_flag, name='deconv2_1')
    #     deconv2_2 = conv_bn_relu(input=deconv2_1, output_chn=64, kernel_size=5,stride=1, use_bias=True, is_training=phase_flag, name='deconv2_2')
    #     print('7',deconv2_2.shape)
    #     deconv3_1 = deconv_bn_relu(input=deconv2_2, output_chn=64, is_training=phase_flag, name='deconv3_1')
    #     deconv3_2 = conv_bn_relu(input=deconv3_1, output_chn=64, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv3_2')
    #     print('8',deconv3_2.shape)
    #     deconv4_1 = deconv_bn_relu(input=deconv3_2, output_chn=32, is_training=phase_flag, name='deconv4_1')
    #     deconv4_2 = conv_bn_relu(input=deconv4_1, output_chn=32, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='deconv4_2')
    #     print('9',deconv4_2.shape)
    #     pred_prob1 = conv_bn_relu(input=deconv4_2, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
    #     pred_prob = conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob')
    #     pred_prob2 = conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob2')
    #     pred_prob3 = conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=5, stride=1, use_bias=True, name='pred_prob3')
    #     print('10',pred_prob.shape)
    #     soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
    #     print('11',soft_prob.shape)
    #     task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
    #     print('12',task0_label.shape)
    #     return  soft_prob,task0_label

    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        print('0',inputI.shape)
        conv1_1 = conv_bn_relu(input=inputI, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv1_1')
        conv1_2 = conv_bn_relu(input=conv1_1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv1_2')
        print('1',conv1_2.shape)
        max1 = tf.nn.max_pool3d(conv1_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv2_1 = conv_bn_relu(input=max1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv2_1')
        conv2_2 = conv_bn_relu(input=conv2_1, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv2_2')
        print('2',conv2_2.shape)
        max2 = tf.nn.max_pool3d(conv2_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv3_1 = conv_bn_relu(input=max2, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv3_1')
        conv3_2 = conv_bn_relu(input=conv3_1, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv3_2')
        print('3',conv3_2.shape)

        max3 = tf.nn.max_pool3d(conv3_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv4_1 = conv_bn_relu(input=max3, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv4_1')
        conv4_2 = conv_bn_relu(input=conv4_1, output_chn=512, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv4_2')
        print('4',conv4_2.shape)


        deconv0_1 = deconv_bn_relu(input=conv4_2, output_chn=512, is_training=phase_flag, name='deconv0_1')
        deconv0_2 = tf.concat([deconv0_1, conv3_2], 4)
        deconv0_3 = conv_bn_relu(input=deconv0_2, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv0_3')
        deconv0_4 = conv_bn_relu(input=deconv0_3, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv0_4')
        print('6',deconv0_4.shape)
        deconv1_1 = deconv_bn_relu(input=deconv0_4, output_chn=256, is_training=phase_flag, name='deconv1_1')
        deconv1_2 = tf.concat([deconv1_1, conv2_2], 4)
        deconv1_3 = conv_bn_relu(input=deconv1_2, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_3')
        deconv1_4 = conv_bn_relu(input=deconv1_3, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_4')
        print('7',deconv1_4.shape)
        deconv2_1 = deconv_bn_relu(input=deconv1_4, output_chn=128, is_training=phase_flag, name='deconv2_1')
        deconv2_2 = tf.concat([deconv2_1, conv1_2], 4)
        deconv2_3 = conv_bn_relu(input=deconv2_2, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv2_3')
        deconv2_4 = conv_bn_relu(input=deconv2_3, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv2_4')
        print('8',deconv2_4.shape)


        pred_prob1 = conv_bn_relu(input=deconv2_4, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob2')
        pred_prob3 = conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob3')
        print('8',pred_prob.shape)
        soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
        print('9',soft_prob.shape)
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        print('10',task0_label.shape)
        return  soft_prob,task0_label

    def train(self):
        print('training the n1 model...')
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        loss_summary_0 =tf.summary.scalar('dice loss',self.Loss)
        self.sess.run(init_op)
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter=1
        data_list =glob('{}/*.nrrd'.format(self.train_data_dir))
        label_list=glob('{}/*.nrrd'.format(self.train_label_dir))
        i=0
        print('valid set: ', self.valid_set)
        for epoch in np.arange(self.epoch):
            i=i+1
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1,hd,hl = load_batch_pair(data_list,label_list,self.train_set)
            print('epoch:',i )
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            #-----------------------
            print('train_output0:',train_output0.shape)
            #-----------------------
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            print('current training loss:',cur_train_loss)
            print('accuracy:', self.accuracy(train_output0, batch_label1.reshape((1,128,128,64))))
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)
        
        self.valid()

    #-------------------------------
    def continue_train(self, x, valid):
        print('training the n1 model...')
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        loss_summary_0 =tf.summary.scalar('dice loss',self.Loss)
        self.sess.run(init_op)
        if self.load_chkpoint(self.chkpoint_dir):
            print(" *****Successfully load the checkpoint**********")
        else:
            print("*******Fail to load the checkpoint***************")
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.valid_set = valid
        self.train_set = {i for i in range(100)} - valid
        counter=1
        data_list =glob('{}/*.nrrd'.format(self.train_data_dir))
        label_list=glob('{}/*.nrrd'.format(self.train_label_dir))
        i=x
        for epoch in np.arange(self.epoch - x):
            i=i+1
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1,hd,hl = load_batch_pair(data_list,label_list,self.train_set)
            print('epoch:',i )
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            #-----------------------
            print('train_output0:',train_output0.shape)
            #-----------------------
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            print('current training loss:',cur_train_loss)
            print('accuracy:', self.accuracy(train_output0, batch_label1.reshape((1,128,128,64))))
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)
        
        self.valid()

    #-------------------------------


    def test(self):
        print('testing patch based model...')  
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load_chkpoint(self.chkpoint_dir):
            print(" *****Successfully load the checkpoint**********")
        else:
            print("*******Fail to load the checkpoint***************")

        pair_list=glob('{}/*.nrrd'.format(self.test_data_dir))
        k=1
        for i in range(len(pair_list)):
            print('generating result for test sample',k)
            test_input,header=load_batch_pair_test(pair_list,i)
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input})
            #implants_post_processed=post_processing(test_output[0,:,:,:])
            #---------------
            if i < 10:
                filename=self.save_dir+r"\implants00%d.nrrd"%i
            elif i < 100:
                filename=self.save_dir+r"\implants0%d.nrrd"%i
            #---------------
            
            nrrd.write(filename,test_output[0,:,:,:].astype('float32'),header)
            k+=1

    

    #-----------------------------------------
    def valid(self):
        print("******************Initiate validation******************")
        data_list =glob('{}/*.nrrd'.format(self.train_data_dir))
        label_list=glob('{}/*.nrrd'.format(self.train_label_dir))
        a = 0
        for i in range(len(self.valid_set)):
            valid_input,valid_label,hd,hl = load_batch_pair_valid(data_list, label_list, list(self.valid_set)[i])
            valid_output = self.sess.run(self.task0_label, feed_dict={self.input_I: valid_input})
            a += self.accuracy(valid_output, valid_label.reshape((1,128,128,64)))
        print("accuracy:",a/10)
    #-----------------------------------------
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s" % ('n1_ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)



    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('n1_ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def accuracy(self, data_defected, label):

        predict = data_defected - label
        predict[predict != 0] = 1
        label[label != 0] = 1
        data_defected[data_defected != 0] = 1
        
        return  (1 - predict.sum()/(label.sum() + data_defected.sum()))


