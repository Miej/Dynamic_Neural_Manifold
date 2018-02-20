# Dynamic Neural Manifold --or-- Dynamic Neural Networks in Tensorflow

Author: Mark Woods ( mark.woods89@gmail.com )

release date: 2/19/18

[![Dynamic Neural Manifold - toy dataset](http://i.imgur.com/M4v5HUb.png)](https://youtu.be/DD2QYcanyc4 "Dynamic Neural Manifold - toy dataset")

In this project, I've built a neural network architecture with a static execution graph that acts as a dynamic neural network in which connections between various neurons are controlled by the network itself. This is accomplished by manipulating the adjacency matrix representation of the network on a per-neuron basis with cell elements representing a 'distance', and masking off connections that are within a threshold. Including a loss term based on the networks sparsity or processing time allows the architecture to optimize its structure for accuracy or speed.

## Intro
Alright, so hopefully I've caught your attention with the title.  To begin, I'd like to explain a little behind why I've created this.

My educational background is actually in the sciences, just at the junction between chemistry and physics.  I went to grad school for theoretical chemistry (quantum mechanics stuff), but I dropped out after a year for various reasons, and switched my focus to the world of data science / machine learning.  Getting started with neural networks, one of the first questions on my mind was (understandably, I think) "How do you design a neural network that is successful at a particular task?".  What surprised me though, was that anywhere I asked that question, from forums, to ML chat rooms, to tech industry professionals, I seemed to be met with the same answer time and time again, which was: "Well, first you start with a neural net that already works..."

As someone who enjoys understanding things from a first-principles sort of basis, that was just about the most unsatisfying answer possible.  It was sort of like hearing, "well, if you want to make a hammer, first start with a hammer..."

So in an effort to deal with that, I wanted to find a somewhat reasonable/reliable way to develop neural networks from the ground up, and try to figure out a vaguely generalizeable solution to the question of how they should be structured. This project is something I made to try to help answer that question for myself, and maybe someone else out there will find it interesting, and perhaps if I'm especially lucky, useful.  So without further ado, let me present my work on Dynamic Neural Networks in Tensorflow - or as I like to refer to it: Dynamic Neural Manifolds.

About the name - Dynamic Neural Manifold.  Dynamic, because it modifies its own structure.  Neural, because it's a neural network.  Manifold, because its structue is essentially defined by the topology of the neurons and their connections.  

## Basics

So in order to describe what I've built here, it's first important to have a basic understanding of what a neural network is and how it works.  But honestly, there are about a thousand websites out there that explain it a lot better than I have time to right now, and with much prettier graphics - so I'll instead assume you have at least a basic/moderate level of understanding.  Essentially though, there are layers of neurons, and predictions are formed via a feed-forward process, after which, the errors of the predictions are used to adjust the network's parameters via the back-propagation algorithm in order to incrementially improve the accuracy of the network's outputs.

So hopefully if that last sentence was boring to you, right now the thing you're most interested in is how exactly I'm claiming to have developed a _dynamic_ neural network architecture in Tensorflow, which is explicitly designed to only handle static computational graphs.  And the answer is: with a little bit of trickery.


## Network Structure

So I'll preface this by saying that there are several reasons why one might not want to use this sort of structure for production use-cases, and that's mostly due to memory limitations and computation speed.  It's still a work in progress, I lack a de juro educational background in computer science, and this is mostly meant to serve as something to help provoke some new ideas/thoughts.  Caveat aside, let's get started.


To start off, we're going to think of our neural network as a graph structure.  And instead of grouping neurons into layers, we're going to consider each single neuron separately.

Now, ignoring things like recurrent neural networks for the time being, the neural network architecture is meant to run in a single direction, and will be prohibited from forming any cyclic connections.  Or in other words, we're specifying that the neural network is a Directed Acyclic Graph (DAG).  

DAGs are pretty cool.  Notably (and most usefully), one of the properties of a Directed Acyclic Graph is that if we consider the adjacency matrix of a DAG, it will only contain elements in the lower triangular.  This also means that if we start with an arbitrary adjacency matrix, if we see that there are no entries in the upper triangular (or diagonal), it can represent a DAG.

![Image of DAG](https://qph.ec.quoracdn.net/main-qimg-02323e19e4c34d26d404bfd0b84a8262)

_Above: an example of a maximally-connected DAG and its corresponding adjacency matrix_

In this case, we are considering each individual neuron in our neural network to occupy a single index in our adjacency matrix representation.  DAGs with only 0 or 1 values are very useful for defining whether there is a connection between two nodes (or in this case, neurons), but I was interested in having the neural network be able to modify its own structure, and having a flat function with a single step is not particularly conducive to generating smooth gradients.  So instead, (and admittedly a little inspired by the wrinkly structure of the brain), I choose to rather define the absolute value of the elements in the adjacency matrix within the range [0,1], and consider it to describe a measure of "closeness", such that a value of 1 would represent two nodes (neurons) that have zero distance between each other, and a value of 0 would represent two neurons with infinite distance between each other.  This is helpful, but not sufficient.

Another important consideration is that we want to find a way to enforce sparsity in our neural network in order to (ideally) limit the computational complexity, so along with our analog 'closeness' values in the adjacency matrix, we will also define a global threshold or cutoff value, below which we consider any 'closeness' values to be essentially zero.

Now let's be a little more specific about the neurons we are using.  Our network structure will contain three separate "types" of neurons:  input neurons, output neurons, and computational neurons.  And we will subject them to the following constraints -

No input neuron connects to another input neuron. Rather, they will simply feed a data point deeper into the network.

No output neuron connects to another output neuron.  Rather, they will simply produce the final output predictions of our network.

With those constraints, we've limited the number of cells in our adjacency matrix representation that we need to consider just a little more, and we are left with a total number of cells equal to: (n_neurons^2 - n_inputs^2 - n_outputs^2 - n_computational) / 2

![Image of our neural net](https://i.imgur.com/5YlDbiw.png)
_Above: The general regions of the neural net's adjacency matrix representation.  Note- input and output regions are also unused._


Next, we're going to define a function for each non-input neuron, and apply it to the adjacency matrix row-wise.  This is a little arbitrary.  Essentially, the closer two connected neurons are, the greater the signal that would otherwise pass between them.  In my code, I've opted to use a function of the form of a sin function raised to some power, and multiplied by a gaussian function.  The goal is simply to create a sort of holography or dimensionality reduction of the n^2 cells in the adjacency matrix.  That will represent our 'closeness' values, which we will then consider to be proportional to a "strength", and then multiply that value by the typical sigmoid(wx+b) activation of each neuron.  I'll explain the rationale for that choice of function.

![closeness functions](https://i.imgur.com/ILRrm9p.png)
*Above: The general form of the "Closeness" function*

![closeness functions](https://i.imgur.com/ONBIKSs.png)
*Above: The "Closeness" functions applied neuron-wise to the adjacency matrix, with random noise in the parameters*

![closeness functions](https://i.imgur.com/MVsgJPX.png)
*Above: The "Closeness" functions applied neuron-wise to the adjacency matrix, with unused regions trimmed out*

A huge use case for neural networks is processing images.  But if our adjacency matrix is in a way defining the relationship between specific-index data points, then the best we could do with just gaussian functions is to describe 1-dimensional images.  However, if we then include a sin component, we can set the period of the sin function to be the width of the given images, and we will essentially be left with a (not quite perfect, but usable) 2-dimensional gaussian function for a given point in an image.  Then, raising the sin fucntion to some power just allows us (or rather, the network) to tweak the curvature of the peaks.  So now we can feasibly have neurons that are specifc to particular regions in a given image, decreasing the magnitude of their inputs as one looks radially further from a given point.

Ok, so good so far.  We have arbitrary functions applied to our adjacency matrix, and we're applying a cutoff value, below which, we simply clip to zero.  Next, this might be unnecessary, but I've somewhat arbitrarily opted to rescale the remaining bits of the function such that the minimum value they hit is not the cutoff value, but rather zero.  I think that seems cleaner, though I haven't investigated if this is a good or a bad choice.  So we're going to inform tensorflow that each of the parameters in these functions are trainable parameters, which will then allow the network to exert control over its own structure.  And because of the DAG representation we're using here, we technically maintain a static computational graph, while de facto creating a fully dynamic neural network where connections between neurons can be created or destroyed, subject to the constraint that we must define some maximum number of neurons the the network is allowed to use.

There's a big caveat here - this choice of structure will undoubtedly limit the time efficiency of the neural network, since technically the output of each neuron can only be evaluated once each neuron before it has been evaluated.  I imagine there's probably a tricky way to optimize this and parallelize a good amount of the computation, progressively propagating the deltas caused by the sequential nature of the DAG representation, but I haven't built that yet.

Anyways, we're not quite done yet, because we've introduced a very particular sort of problem into our network now; did you catch it?  Congrats if you did, because I didn't at first.  I only discovered it when I tried to run this architecture.

Yes, technically the network can indeed modify its own structure now, but since we've not limited that in any way, that means the network can now "accidentally" lobotimize itself, by severing enough of its neuronal connections so as to prevent any end-to-end input-to-output connections from existing.  This is bad news for trying to do forward propagation, and this is bad news for trying to do backpropagation.  So we will need to be a little tricky once again.

## Loss functions

The way we're going to deal with the lobotimization problem is actually somewhat simple.  Thinking back to the nature of the problem, the issue is that the network can now lose end-to-end connectivity.  An analogous way to think about it is that we essentially have a power line, and the metal wire inside the plastic shielding can break - meaning we need to be able to detect when it's at risk of doing so.  So taking a page from the universe's notebook, we're going to implement a term in our loss function to penalize deviations from the conservation of energy.  Or rather in this case, flow.  Basically, what goes in, must come out.  One option would be to enforce this on the level of individual neurons, summing the magnitude of the "closeness"/"strength" function values for each incoming connection, and subtract that from the sum of the "closeness"/"strength" function values for each outgoing connection.  But that would be a little more tedious than we actually need, since it double-compute values that would cancel out anyways.  

Instead, we can look back at our adjacency matrix representation, and consider the three separate chunks of neurons: inputs, outputs, and computational neurons.  Instead of enforcing the flow conservation on an individual level, we can enforce it on a much larger level, and it will behave the same.  So in other words, we just need to sum the "strength" of each connection starting from an input node and terminating in a computational node, and subtract from that the sum of the "strength" of each connection starting from a computational node and terminating in an output node.  That way, if a substantial difference begins to form there, the power line's "cable" will begin to grow "hot" (ie: the loss term describing the delta will increase), and the network will know that it needs to attend to that.  Of course, if your learning rate is too high and your network is especially small, its probably possible for the network to just explosively lobotimize itself in a single step anyways.  For all reasonable purposes though, this solution seems to work well.

![flow conservation](https://i.imgur.com/mOhH218.png)
*Above: The general scheme of the flow regularization.  Yellow regions cancel out.*

We're going to make this more interesting now.  Remember how we introduced a cutoff threshold?  So as to introduce sparsity into the network?  Well now that we have our dynamic network structure set up, we can get creative.  What sort of network would you design if lives were on the line, and you needed to be completely confident that a given prediction was correct, and you dont care how long it takes to get that answer?  What sort of network would you design if you needed predictions that only had to be "good enough", but you needed them _fast_.  There is an obvious tradeoff between accuracy and speed.  

Time for our next caveat.  Tensorflow currently has several sparse tensor methods, however, there are several specific ops that as of yet do not support gradients, despite their dense counterparts successfully doing so.  Trust me, I built out the entire sparse implementation of this network, only to spend a week or two tracking down the reason why despite all efforts, it still wouldn't work.  I'm looking at you, tf.sparse_concat() . Caveat over.

So even though we can't yet get as tricky as we want to with our next loss term, we can still approximate it.  In a perfect world, Tensorflow's sparse tensor ops would support gradients, and we would measure the amount of time it takes to process a single training step (via time.process_time() ), and use that to build another term in our loss function.  The result of this would be that the network would be able to adjust its own structure _based on the hardware it is being run on_ in order to optimize the mentioned trade-off between speed and accuracy.  But since that's not available yet (and I haven't built it into the source code yet), we're going to make an approximation so that we can _simulate_ what sort of effects it would have.  To do that, we're simply going to count the number of elements in our "closeness" adjacency matrix with non-zero values and divide that by the total number of elements available, thus producing a value representing how sparsely populated the DAG neural net is as a percentage.  Explicitly, the approximation we're making is that in a fully-sparse implementation, there would be a linear relationship between the sparsity of the adjacency matrix and the time required to process a single step, which is almost certainly false, but useful enough for now.

Last but not least, we do need to include a term for the actual proper loss function we want to use for comparing the predictions of the network to the objectives, be that a MSE, or a cross entropy, or whatever.  

So finally we have our loss function, which has three terms: the objective loss, the flow loss, and the sparsity loss.  And in order to tweak the balance between (simulated) speed and accuracy, we can simply multiply the three terms by different, user-defined coefficients in order to represent the relative "importance" of each one.  The objective loss represents the importance of the accuracy, the sparsity loss represents the importance of the (simulated) speed, and I think the flow loss is best understood to be how much flexibility the network is given in terms of adjusting its structure (ie: too low and the net might lobotimize itself, too high and it simply wont pay attention to learning the objective)

And that's about it.  If you've followed along, you should now have an understanding the general concepts that drive this network architecture.  I've gone ahead and included a few examples of the network processing simple datasets in order to demonstrate that it is in fact capable of creating performant models, even though I recognize that it does not necessarily come close to the current state-of-the-art in terms of either speed nor accuracy.

## Notes

- It is in fact possible to still use a time.process_time() based loss even with the dense tensor implementation, and perhaps it might even lead to some interesting results along the lines of this (very cool) article: https://www.damninteresting.com/on-the-origin-of-circuits/  . That said, I haven't observed particularly interesting results from using it yet myself, except that comparing the % sparsity loss to the time loss reveals some structure, which allows us to speculate at optimizations in the source code for processing the network's calculations


- These sort of models aren't necessarily meant to serve as state-of-the-art production-status implementations, since as I mentioned a few times, they have some significant limitations in terms of space/time efficiency.  But that said, I do find the architecture to be an interesting way of creating a somewhat generalized approach to applying neural networks to arbitrary problem domains.  Additionally, due to its nature, it is capable of defining network structures of far greater complexity that would be feasible to do by hand with a layer-based approach.  Obviously, this doesn't really do us any favors in terms of interpreting the structures present in the network, but then again...that's sort of fun in its own way.


- The neuron-wise 'closeness' functions don't seem to adjust the value of the mean of the gaussian part very much (ie: shift the functions left or right).  I believe this is because due to the random initialization, the initially activated connections very quickly gain a comparative advantage over any other ones, thus limiting the ability of the network to explore substantially different connectivity patterns than its original initialization.  Perhaps a solution to this would be to simply greatly increase the value of the initialization of the sigma term to a value closer to the total number of neurons, so that the functions do not exhibit large regions in the adjacency matrix representation where all connections in a given direction are below the cutoff threshold.

## Future directions

- Well for one, it would be neat to finally have a working, fully sparse implementation of this in order to be able to utilize a process_time -based loss, instead of just simulating it with a sparsity loss.


- In the provided code, the neurons are the simplest sigmoid(wx+b) feed-forward type.  It would be relatively trivial to convert them into lstm neurons.


- I'm a little curious to see what might happen if we add n additional nodes to both the output and the input, and run the output from those nodes into the input of the next step in an online training setting, and enforcing no loss objective for those specific outputs.  I suppose this would be akin to making a recurrence/memory mechanism on the network-scale, as opposed to on the individual neuron-level?  Although presumably without any loss contributed from those outputs, the network simply wouldnt adjust any parameters related to them - might be pointless.


- I mentioned this earlier, but I'm sure there's some way to improve the performance of the computation of this network structure to help increase the parallelizability of it.


- It would be possible to include a memory-based term in the loss function as well, though this would only be useful if we were using a sparse implementation also I think.

## Closing remarks

I hope you enjoy this work.  It's the culmination of a lot of time, effort, struggle, and difficulty.  If you have enjoyed reading this and/or playing with the architecture, I'd love to hear from you.  Special thanks to the various people who I've bounced ideas off, sought troubleshooting advice from, and rambled incessantly at in the #notpron channel on quakenet irc, as well as the ##machinelearning channel on freenode.  _Very_ special thanks to my brother Mike, who has gone far above and beyond any reasonable expectation in supporting me though my endeavours, and without whom this work simply wouldn't have been possible.

## Some Examples

## Dataset: toy dataset (addition / subtraction)

These results are straight from the included notebook "dynamic_neural_manifold-toy_dataset.ipynb"


This dataset looks like:

a, b, c, d, e = random normally-distributed variables with zero mean, unit variance

data = [[a, b, c, d, e] \* batch_size]

target = [[b+e, e-d, b+e+d, e-d-c] \* batch_size]


\# of neurons: 25

\# of inputs: 5

\# of outputs: 4

\# of computational neurons: 16

batch size (same for both train/test): 50

\# of training steps: 5k

data collection step size: 1

initial cutoff value: sigmoid(-1.0) ~= 0.268

periodicity value: 5 (# of input datapoints)

Objective loss coefficient: 10^4

Flow loss coefficient: 10^0

sparsity loss coefficient: 10^-1

[![Dynamic Neural Manifold - toy dataset](http://i.imgur.com/M4v5HUb.png)](https://youtu.be/DD2QYcanyc4 "Dynamic Neural Manifold - toy dataset")

![Objective loss](https://i.imgur.com/LSXi2vd.png)
*Above: The objective loss (mean squared error) over the training cycle*
    
![Flow loss](https://i.imgur.com/WR2kmeu.png)
*Above: Flow loss of the network over the training cycle*
    
![sparsity loss](https://i.imgur.com/wkBrbXr.png)
*Above: sparsity loss of the network over the training cycle*


##### Notes

- The first datapoint was intentionally not used in the objective to see if the network would learn to ignore any of its contributions.  From the video, it appears to be doing so.


- As evidenced in the flow loss graph, the network is able to temporarily incur additional loss from flow inequality if it means the network will find a more favorable state.


- The sparsity of the network seems to begin too low for the given dataset, leading it to increase its computational complexity(-ish) in order to better optimize the objective loss.

## Dataset: mnist


These results are straight from the included notebook "dynamic_neural_manifold-mnist.ipynb"

\# of neurons: 1000

\# of inputs: 784

\# of outputs: 10

\# of computational neurons: 206

batch size (same for both train/test): 50

\# of training steps: 50k

data collection step size: 100

initial cutoff value: sigmoid(-1.0) ~= 0.268

periodicity value: 28 (width of images)

Objective loss coefficient: 10^3

Flow loss coefficient: 10^-4

sparsity loss coefficient: 10^-1

[![Dynamic Neural Manifold - mnist](http://i.imgur.com/DqVYhFY.png)](https://youtu.be/pGY5WF2VHV8 "Dynamic Neural Manifold - mnist")

![accuracy on test set](https://i.imgur.com/TTp3sL9.png)
_accuracy on test set_

final accuracy: 98%

![sparsity loss term](https://i.imgur.com/swpCH4B.png)
_sparsity loss term_

final sparsity: 3%

##### Notes

- It is hard to get a reasonable idea of the larger-picture dynamics of hundreds of neurons and thousands of connections from a tiny video.


- Early into the training, about a quarter of the way into it, the network exhibits a sudden shift in its connectivity / neuron "distances".  Why?  How?  I don't know.


- Near the end of the training cycle, the accuracy has begun to level off in the realm of high-90% accuracy, while the sparsity loss term is still steadily decreasing.  I think this is neat.  It shows that the network is able to essentially trim neuronal connections that do not contribute significantly to the end-predictions.  Or it is replacing them by building that knowledge into other existing connections.  I don't know.


Cheers.  Feel free to get in contact if you enjoyed this project.  
I am currently seeking employment, and I'd especially love to move back to NYC.  It's an amazing city.

-Mark Woods

mark.woods89@gmail.com

