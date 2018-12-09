import kmodels
from keras import optimizers
import datagen
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


if __name__ == '__main__':
    model = kmodels.densenet()
    model.compile(optimizers.Adam(),
                  'binary_crossentropy'
                  )

    csv = open('train.csv', 'r').readlines()
    lists = []
    for line in csv[1:]:
        id_, target = line[:-1].split(',')
        lists.append((id_, target.split()))
    train_list = lists[:26073]
    test_list = lists[26073:]
    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/HPAIC/train'
    except PermissionError:
        rootpath = '/media/palm/data/Human Protein Atlas/train'
    train_dataset = datagen.Generator(train_list,
                                      rootpath,
                                      28,
                                      (224, 224),
                                      batch_size=32
                                      )
    test_dataset = datagen.Generator(test_list,
                                     rootpath,
                                     28,
                                     (224, 224),
                                     batch_size=32
                                     )
    callback = ModelCheckpoint('weights/dense201-1.h5',
                               save_weights_only=1,
                               save_best_only=1,
                               mode='min'
                               )
    f1 = Metrics()
    model.fit_generator(train_dataset,
                        validation_data=test_dataset,
                        epochs=20,
                        callbacks=[callback],
                        )
