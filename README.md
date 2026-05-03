# Torch Predictive Coding

A simple [predictive coding](https://arxiv.org/pdf/2506.06332) layer on top of [pytorch](https://github.com/pytorch/pytorch)

Currently set up for simple supervised classification tasks. See `examples/cifar10.py` and `examples/mnist.py` for examples of how to do this.

## Planned features
- [x] Export to and from pytorch models (partially supported, needs to be more tightly integrated into library)
- [ ] Show how a model trained on classification can be used generatively
- [ ] Support regression tasks

## General TODO:
- [x] Integrate save and load mechanism into `PCNetwork`
- [ ] Models trained on backprop should be convertible to PC and vice versa
