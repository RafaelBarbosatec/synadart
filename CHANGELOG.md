# Version 0.1.0

- Added Multi-layer Perceptron and a basic algorithm for backpropagation

# Version 0.1.1

- Added README.md, updated formatting

# Version 0.2.0

- Added FF ( feedforward ) and simple Perceptron networks
- Added LReLU, eLU and tanh activation function
- Renamed 'sigmoid' to 'logistic' function

# Version 0.2.1

- Removed FF ( feedforward ) and simple Perceptron networks in favour of an upcoming simpler implementation of basically the same idea, through just one network model.
- Added [learningRate] as a parameter, and removed the hard-coded value of 0.2.
- Organised the files slightly
- Updated documentation of `Neuron`

# Version 0.2.2

- Updated documentation of `Layer` and removed a chunk of dead code.

# Version 0.2.3

- Updated documentation of `Network`
- Replaced `process()` in `Layer` with an `output` getter, simplifying the implementation of getting each `Neuron`'s output