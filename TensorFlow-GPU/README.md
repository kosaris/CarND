## How to install Tensorflow 1.4.1 GPU with CUDA Toolkit 9.1 and cuDNN 7.0.5 for Python 3 on Ubuntu 16.04-64bit 

[The link here](http://www.python36.com/install-tensorflow141-gpu/) is a very good step by step guide on how to do this and pretty much works out of the box. You have to build tensorflow from source.

To test the installation you can run the following code:

```
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

```

if the code runs without an issue, you should see something similar to this:
```
2018-01-14 09:47:36.298943: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-01-14 09:47:36.299202: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.6705
pciBusID: 0000:02:00.0
totalMemory: 10.91GiB freeMemory: 10.35GiB
2018-01-14 09:47:36.299213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2018-01-14 09:47:36.334753: I tensorflow/core/common_runtime/direct_session.cc:299] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1

Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-01-14 09:47:36.335191: I tensorflow/core/common_runtime/placer.cc:874] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
2018-01-14 09:47:36.335203: I tensorflow/core/common_runtime/placer.cc:874] b: (Const)/job:localhost/replica:0/task:0/device:CPU:0
2018-01-14 09:47:36.335208: I tensorflow/core/common_runtime/placer.cc:874] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
[[22. 28.]
 [49. 64.]]

```

You can see that a , b are defined using the cpu but the matmul operation is using the GPU (default device).

## Setting up Conda
Setting up with Conda required me to create my own environment (yml) files. In order to have the correct versions.

After you install Conda, run:

```which python3```

you will get something similar to:

```/home/$USER/miniconda3/bin/python3```

whereas you want to run
```/usr/bin/python3```

I opened python3 from ```/usr/bin/``` location and see that I am running version 3.5.2. Now we can instruct Conda to install this version of python in the environment.

As part of the tensorflow build process, will get a whl file in your tensorflow_pkg package. Conda can be instructed to use that file for installing tensorflow in the environment.

```
name: My-Lab
dependencies:
- python=3.5.2
- numpy==1.14.0
- pip>=8.1.2
- pip:
  - /home/$USER/Downloads/tensorflow-1.4.1/tensorflow_pkg/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl
```
## Training a Network using GPU
If the above steps work, you can run your code (after setting up conda environment and configuring pydev interpreter to use the conda python from the environment).

[example1](example1.py) is an example of how you can switch between using cpu and gpu by changing one line (in fact only one letter).

```with tf.device('/gpu:0'):```

or

```with tf.device('/cpu:0'):```



## Troubleshooting
### No Module Named tensorflow

```ModuleNotFoundError: No module named 'tensorflow'```

The problem was caused because I was running python from the miniconda path. Run from ```/usr/bin/python3```

### libcublas problems
```
libcublas.so.8.0: cannot open shared object file: No such file or directory
```

This problem was caused due to wrong conda environment. Use the whl file created when you build tensorflow from source. Not using the default conda tensorflow versions.

