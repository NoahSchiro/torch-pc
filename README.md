# Torch Predictive Coding

A simple [predictive coding](https://arxiv.org/pdf/2506.06332) layer on top of [pytorch](https://github.com/pytorch/pytorch)

Currently set up for simple supervised classification tasks. See `examples/cifar10.py` and `examples/mnist.py` for examples of how to do this.

## Installation

```
pip install torch-pc
```

## Development

```
uv sync

# Try an example
uv run example/mnist.py
```

## Sample code

`torch-pc` is meant to integrate tightly with `torch` so things such as data loading / data transformations remain the same. Saving and loading a PC model is easy with a simple `save` and `load` method on `PCNetwork`. It is also possible to transfer weights from a backprop model to PC and vice versa (assuming architecture matches).

Because PC is a relatively new method of training models, most research has been conducted around linear layers so all models must use linear layers. Future version will expand on this, but currently model instatiation looks like this:
```
model = PCNetwork(
    dims=[256, 512, 256, 128], # Input dimensions of your linear layers
    output_dim=10,             # Dimensions in your output / prediction
    activation_fn=torch.tanh,  # Type of activation between layers (default: relu)
    activation_deriv=lambda a: 1 - torch.tanh(a)**2, # Derivative of activation function
    use_bias=False, # Optional, adds bias term to each linear layer, default false
)
```

Kick off PC training:
```
energy_history, supervised_energy_history = train_pcn(
    model=model, # PCNetwork
    data_loader=train_dl, # standard torch DataLoader class
    num_epochs=10, 
    eta_infer=0.1,  # Learning rate on inference part of training
    eta_learn=0.01, # Learning rate on update part of training
    infer_steps=10, # Number of inference steps per batch
    T_learn=3,      # Number of learning steps per batch.
    device=DEVICE,
)
```

Run test for classification tasks (returns top-1 and top-3 accuracy):
```
acc1, acc3 = test_pcn_classify(
    model=model,         # PCNetwork
    data_loader=test_dl, # standard torch DataLoader class
    infer_steps=300,     # Number of inference steps per batch
    eta_infer=0.1,       # Learning rate on inference part of prediction
    device=DEVICE,
)
```

Run test for regression tasks (returns mean squared error and mean average error):
```
mse, mae = test_pcn_regress(
    model=model,         # PCNetwork
    data_loader=test_dl, # standard torch DataLoader class
    infer_steps=300,     # Number of inference steps per batch
    eta_infer=0.1,       # Learning rate on inference part of prediction
    device=DEVICE,
)
```
