import tensorflow as tf
import os,sys
from tfr_data_processing import write_tfr, get_tfrs, load_tfrs
indir=sys.argv[1]
filenames = tf.io.gfile.glob(f"{indir}/*.tfrecords")

batch_size = 8
train_data = get_tfrs(filenames, batch_size)
#train_data = train_data.batch(batch_size)

#print(train_data)
for t in train_data.take(2):
    print(t)

#print(train_data)
#train_data = next(iter(train_data))
#print(train_data)
