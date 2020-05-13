# Horovod practice
This repository keeps a simple note and code for distributed training practice using horovod.

## Intro
### Horovod
Horovod is an open source distributed training framework which supports TensorFlow, Keras, PyTorch and MXNet. It implemented with `OpenMPI` & `ring all-reduce algorithm` and was easy to submit jobs on modern supercomputers. The main benifit is it requires only a few lines of modification to the original code.

### Code modification
- Initializing Horovod: `hvd.init()`
- Control the calling process on specified worker (e.g. worker 0) with horovod rank: `if hvd.rank() == 0`
- Adjust learning rate based on number of GPUs and add Horovod distributed optimizer: 
    ```
    opt = optimizers.SGD(0.01 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False 
                  )
    ```
    **Note:** Specify `experimental_run_tf_function=False` to ensure TensorFlow uses hvd.DistributedOptimizer() to compute gradients
- Add callback: 
  ```
  callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
               hvd.callbacks.MetricAverageCallback(),
               hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1)]
  ```
    - broadcast initial variable states from rank 0 to all other processes.
    - average metrics among workers at the end of every epoch.
    - using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
- Save checkpoints only on worker 0 to prevent other workers from corrupting them.
  ```
  if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint.h5', save_best_only=True, save_weights_only=False))
  ```
### Run Horovod
**(To be complete 1)**
## Setup
The following steps demonstrated how to setup a working environment for multi-GPU training with Horovod.
### Prerequisite
- Hardware: A machine with at least two GPUs or using Qemu virtual machine for toy experiment.
- Software: Open MPI, Horovod, DL framwork (TensorFlow, Keras, PyTorch or MXNet)
### Installation
**(To be complete 2)**
