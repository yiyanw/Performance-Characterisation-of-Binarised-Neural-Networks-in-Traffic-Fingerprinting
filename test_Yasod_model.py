import numpy as np
import tensorflow as tf
from IoT.utility import LoadDataIot, LoadDataIotNumeric, LoadDataIotNumeric_v2, LoadDataIotNumeric_v3

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Yasod_model/tf_lite_fullint_softmax.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# Test model on random input data.
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIot()
# X_train = np.reshape(X_train, (X_train.shape[0], -1, 1))
X_train = np.reshape(X_train[0], (1, -1, 1))
X_train = X_train.astype('int8')
input_shape = input_details[0]['shape']
input_data = np.array(X_train, dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


print(np,max(y_train))