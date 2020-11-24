from glob import glob
from conv3 import *
import numpy as np
import nrrd
from data_loader import *
import os
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
        self.inputI_chn     = 1
        self.output_chn     = 2
        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 10000
        self.model_name     = 'n2.model'
        self.save_intval    = 100
        self.build_model()
        self.init_train_set()

        # directory where the checkpoint can be saved/loaded
        self.chkpoint_dir   = r"D:\autoimplant-master\ckpt"
        # directory containing the 100 training defective skulls
        self.train_data_dir = r"D:\autoimplant-master\training_set\defective_skull_p2"
        # ground truth (implants) for the training data
        self.train_label_dir = r"D:\autoimplant-master\training_set\implant_p2"
        # test data directory
        self.test_data_dir = r"D:\autoimplant-master\test_set_for_participants"
        # directory where the predicted implants from model n1 is stored 
        self.bbox_dir = r"D:\autoimplant-master\predictions_n1"
        # where to save the predicted implants
        self.save_dir = r"D:\autoimplant-master\predictions_n2"



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
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,256,256,128, self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,256,256,128,1], name='target')
        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        #3D voxel-wise dice loss
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        #self.main_softmax_loss=self.softmax_crossentropy_loss(self.soft_prob, self.input_gt[:,:,:,:,0])
        # final total loss
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        # create model saver
        self.saver = tf.train.Saver()

    #-------------------
    def init_train_set(self):
        r = {i for i in range(600)}
        #self.valid_set = {random.randint(0, 99) for _ in range(10)}
        self.valid_set = set()
        while(len(self.valid_set) < 120):
            rand = random.randint(0, 599)
            if(not (rand in self.valid_set)):
                self.valid_set.add(rand)
        self.train_set = r - self.valid_set
    #-------------------



    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        print('0',inputI.shape)
        conv1_1 = conv_bn_relu(input=inputI, output_chn=8, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv1_1')
        conv1_2 = conv_bn_relu(input=conv1_1, output_chn=8, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv1_2')
        print('1',conv1_2.shape)
        max1 = tf.nn.max_pool3d(conv1_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv2_1 = conv_bn_relu(input=max1, output_chn=16, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv2_1')
        conv2_2 = conv_bn_relu(input=conv2_1, output_chn=16, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv2_2')
        print('2',conv2_2.shape)
        max2 = tf.nn.max_pool3d(conv2_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv3_1 = conv_bn_relu(input=max2, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv3_1')
        conv3_2 = conv_bn_relu(input=conv3_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv3_2')
        print('3',conv3_2.shape)
        max3 = tf.nn.max_pool3d(conv3_2, [1,2,2,2,1], [1,2,2,2,1], padding='VALID')
        conv4_1 = conv_bn_relu(input=max3, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv4_1')
        conv4_2 = conv_bn_relu(input=conv4_1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv4_2')
        print('4',conv4_2.shape)
        deconv0_1 = deconv_bn_relu(input=conv4_2, output_chn=64, is_training=phase_flag, name='deconv0_1')
        # deconv0_2 = tf.concat([deconv0_1, conv3_2], 4)
        deconv0_3 = conv_bn_relu(input=tf.concat([deconv0_1, conv3_2], 4), output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv0_3')
        deconv0_4 = conv_bn_relu(input=deconv0_3, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv0_4')
        print('5',deconv0_4.shape)
        deconv1_1 = deconv_bn_relu(input=deconv0_4, output_chn=32, is_training=phase_flag, name='deconv1_1')
        # deconv1_2 = tf.concat([deconv1_1, conv2_2], 4)
        deconv1_3 = conv_bn_relu(input=tf.concat([deconv1_1, conv2_2], 4), output_chn=16, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_3')
        deconv1_4 = conv_bn_relu(input=deconv1_3, output_chn=16, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_4')
        print('6',deconv1_4.shape)
        deconv2_1 = deconv_bn_relu(input=deconv1_4, output_chn=8, is_training=phase_flag, name='deconv2_1')
        # deconv2_2 = tf.concat([deconv2_1, conv1_2], 4)
        deconv2_3 = conv_bn_relu(input=tf.concat([deconv2_1, conv1_2], 4), output_chn=8, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv2_3')
        print('7',deconv2_3.shape)
        pred_prob1 = conv_bn_relu(input=deconv2_3, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob2')
        
        soft_prob=tf.nn.softmax(pred_prob2,name='task_0')
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        return soft_prob,task0_label

    def train(self):
        print('training the n2 model')
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        loss_summary_0 =tf.summary.scalar('dice loss',self.Loss)
        self.sess.run(init_op)
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter=1
        data_list =sort(glob('{}/*.nrrd'.format(self.train_data_dir)))
        label_list=sort(glob('{}/*.nrrd'.format(self.train_label_dir)))
        bbox_list=sort(glob('{}/*.nrrd'.format(self.bbox_dir)))
        i=0
        print('valid set: ', self.valid_set)
        for epoch in np.arange(self.epoch):
            i=i+1
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1,hd,hl= load_bbox_pair(bbox_list,data_list,label_list,self.train_set)
            print('epoch:',i )
            
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            print('current training loss:',cur_train_loss)
            print('accuracy:', self.accuracy(train_output0, batch_label1.reshape((1,256,256,128))))
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)
        self.valid()

        #-------------------------------
    def continue_train(self, x, valid):
        print('training the n2 model...')
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
        self.train_set = {i for i in range(600)} - valid
        counter=x
        data_list =sort(glob('{}/*.nrrd'.format(self.train_data_dir)))
        label_list=sort(glob('{}/*.nrrd'.format(self.train_label_dir)))
        bbox_list=sort(glob('{}/*.nrrd'.format(self.bbox_dir)))
        i=x
        for epoch in np.arange(self.epoch - x):
            i=i+1
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1,hd,hl= load_bbox_pair(bbox_list,data_list,label_list,self.train_set)
            print('epoch:',i )
            
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            print('current training loss:',cur_train_loss)
            print('accuracy:', self.accuracy(train_output0, batch_label1.reshape((1,256,256,128))))
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
        data_list =sort(glob('{}/*.nrrd'.format(self.test_data_dir)))
        bbox_list=sort(glob('{}/*.nrrd'.format(self.bbox_dir)))
        
        k=1
        for i in range(len(data_list)):
            print('generating result for test sample',k)
            test_input,header=load_bbox_pair_test(bbox_list,data_list,i)
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input})
            #implants_post_processed=post_processing(test_output[0,:,:,:])
            #filename=self.save_dir+"implants%d.nrrd"%i
            filename=self.save_dir+bbox_list[i][-15:-5]+'.nrrd'
            nrrd.write(filename,test_output[0,:,:,:].astype('float32'),header)
            k+=1
    
    def valid(self):
        print("******************Initiate validation******************")
        data_list =sort(glob('{}/*.nrrd'.format(self.train_data_dir)))
        label_list=sort(glob('{}/*.nrrd'.format(self.train_label_dir)))
        bbox_list=sort(glob('{}/*.nrrd'.format(self.bbox_dir)))
        a = 0
        for i in range(len(self.valid_set)):
            valid_input,valid_label,hd,hl = load_bbox_pair_valid(bbox_list, data_list, label_list, list(self.valid_set)[i])
            valid_output = self.sess.run(self.task0_label, feed_dict={self.input_I: valid_input})
            a += self.accuracy(valid_output, valid_label.reshape((1,256,256,128)))
        print("accuracy:",a/10)


    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s" % ('n2_ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('n2_ckpt')
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







