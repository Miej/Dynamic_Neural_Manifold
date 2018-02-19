# Dynamic Neural Manifold

In this project, I've built a neural network architecture with a static execution graph that acts as a dynamic neural network, in which connections between various neurons are controlled by the network itself.  This is accomplished by manipulating the adjacency matrix representation of the network on a per-neuron basis with cell elements representing a 'distance', and masking off connections that are within a threshold. Including a loss term based on the networks sparcity or processing time allows the architecture to optimize its structure for accuracy or speed.


Please reference the included readme notebook for extended description and examples.


