# -*- coding: utf-8 -*-
import tensorflow as tf

import face
import read_data

checkpoint_dir = 'kaggle/'

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = face.model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)    
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('error')
    '''
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.")
    '''

    X, y = read_data.input_data(test=True)
    y_pred = []

    y_batch = y_conv.eval(feed_dict={face.x: X, face.keep_prob: 1.0})
    y_pred.extend(y_batch)
    print (y_pred)
    print ('predict test image done!')

    output_file = open('kaggle/submit3.csv', 'w')
    output_file.write('RowId,Location\n')

    IdLookupTable = open('kaggle/IdLookupTable.csv')
    IdLookupTable.readline()

    for line in IdLookupTable:
        RowId, ImageId, FeatureName = line.rstrip().split(',')
        image_index = int(ImageId) - 1
        feature_index = read_data.keypoint_index[FeatureName]
        feature_location = y_pred[image_index][feature_index] * 96
        output_file.write('{0},{1}\n'.format(RowId, feature_location))

    output_file.close()
    IdLookupTable.close()
    