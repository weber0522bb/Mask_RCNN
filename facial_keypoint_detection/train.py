# -*- coding: utf-8 -*-
import random
import tensorflow as tf

import face
import read_data


def save_model(saver, sess, save_path):
    path = saver.save(sess, save_path)
    print ('model save in :{0}'.format(path))

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = face.model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

    #变量初始化
    sess.run(tf.initialize_all_variables())
    X,y = read_data.input_data()
    X_valid, y_valid = X[:read_data.VALIDATION_SIZE], y[:read_data.VALIDATION_SIZE]
    X_train, y_train = X[read_data.VALIDATION_SIZE:], y[read_data.VALIDATION_SIZE:]

    best_validation_loss = 1000000.0
    current_epoch = 0
    TRAIN_SIZE = X_train.shape[0]
    train_index = list(range(TRAIN_SIZE))
    random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    saver = tf.train.Saver()

    print ('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
    for i in range(read_data.EPOCHS):
        random.shuffle(train_index)  # 每个epoch都shuffle一下效果更好
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in range(0, TRAIN_SIZE, read_data.BATCH_SIZE):
            print ('epoch {0}, train {1} samples done...'.format(i, j))

            train_step.run(feed_dict={face.x: X_train[j:j + read_data.BATCH_SIZE],
                                      face.y_: y_train[j:j + read_data.BATCH_SIZE], face.keep_prob: 0.5})

        # train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
        validation_loss = rmse.eval(feed_dict={face.x: X_valid, face.y_: y_valid, face.keep_prob: 1.0})

        print ('epoch {0} done! validation loss:{1}'.format(i, validation_loss * 96.0))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            save_model(saver, sess, read_data.SAVE_PATH)  # 即时保存最好的结果
        elif (i - current_epoch) >= read_data.EARLY_STOP_PATIENCE:
            print ('early stopping')
            break

    X, y = read_data.input_data(test=True)
    y_pred = []

    TEST_SIZE = X.shape[0]
    for j in range(0, TEST_SIZE, read_data.BATCH_SIZE):
        y_batch = y_conv.eval(feed_dict={face.x: X[j:j + read_data.BATCH_SIZE], face.keep_prob: 1.0})
        y_pred.extend(y_batch)

    print ('predict test image done!')

    output_file = open('kaggle/submit.csv', 'w')
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
    
