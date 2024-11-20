# MiniTorch Module 3

## Graph for 3.4
Here is my graph comparing naive operations vs. GPU operation on matrix multiplication.
<img src="./3.4_timing_graph.png?raw=true" alt="Timing graph" width="400"/>


## Plots for Simple Dataset

Results from GPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 100
* size of hidden layer = 100

<img src="./simple_gpu.png?raw=true" alt="Simple GPU Plot" width="400"/>


Results from CPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 100
* size of hidden layer = 100

<img src="./simple_cpu.png?raw=true" alt="Simple CPU Plot" width="400"/>

## Plots for Split Dataset

Results from GPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 300
* size of hidden layer = 100

Ed post #413 says full credit will be awarded for achieving 48 correct classifcations or above.
<img src="./split_gpu.png?raw=true" alt="Split GPU Plot" width="400"/>


Results from CPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 300
* size of hidden layer = 100

<img src="./split_cpu.png?raw=true" alt="Split CPU Plot" width="400"/>

## Plots for XOR Dataset

Results from GPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 300
* size of hidden layer = 100


Ed post #413 says full credit will be awarded for achieving 48 correct classifcations or above.
<img src="./xor_gpu.png?raw=true" alt="XOR GPU Plot" width="400"/>


Results from CPU training

Hyperparameters are:
* learning rate = 0.01
* number of epochs = 300
* size of hidden layer = 100

Ed post #413 says full credit will be awarded for achieving 48 correct classifcations or above.
<img src="./xor_cpu.png?raw=true" alt="XOR CPU Plot" width="400"/>


## Timing on a larger model
With following hyperparameters I achieved the following results with GPU and CPU implementations for the simple dataset
* learning rate = 0.01
* number of epochs = 100
* size of hidden layer = 200

GPU time per epoch

<img src="./gpu_big.png?raw=true" alt="gpu_big Plot" width="400"/>

CPU time per epoch

<img src="./cpu_big.png?raw=true" alt="cpu_big Plot" width="400"/>



#

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py