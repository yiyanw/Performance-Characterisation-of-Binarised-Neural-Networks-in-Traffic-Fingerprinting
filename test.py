import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True)).run(tf.constant(1) + tf.constant(1))
