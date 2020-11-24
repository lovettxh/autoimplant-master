import os

import tensorflow as tf
from n1_model import auto_encoder

#from n2_model import auto_encoder
#from skull_completion_model import auto_encoder

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth=True
sess1 = tf.Session(config=config)


with sess1.as_default():
    with sess1.graph.as_default():
        model = auto_encoder(sess1)
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('trainable params:',total_parameters)
        # to train the selected model
        v={512, 0, 513, 3, 517, 8, 523, 524, 526, 529, 19, 531,
        25, 27, 540, 30, 544, 545, 548, 37, 551, 41, 560, 51, 
        566, 56, 63, 66, 67, 580, 69, 71, 587, 76, 77, 80, 592, 
        596, 85, 89, 91, 95, 96, 97, 118, 121, 125, 132, 137, 
        142, 154, 158, 160, 168, 172, 173, 177, 183, 184, 187, 189, 
        191, 194, 198, 199, 202, 205, 210, 214, 219, 227, 236, 238, 
        239, 246, 251, 254, 264, 271, 273, 275, 280, 283, 306, 313, 
        325, 326, 327, 331, 349, 352, 360, 369, 375, 377, 382, 385, 
        386, 389, 391, 398, 412, 425, 436, 443, 450, 455, 458, 462, 
        474, 477, 482, 484, 485, 487, 495, 498, 503, 510}
        #model.train()
        #model.test()
        model.continue_train(28500, v)
        # to generate implants using the trained model
        #print(model.accuracy())


 

