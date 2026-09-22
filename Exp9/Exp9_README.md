# Experiment 9 — Perceptron (PLA) vs. Multilayer Perceptron

## Objective
Understand the single-layer Perceptron as the foundational neural learning
algorithm, its theoretical limitations, and how stacking layers with non-linear
activation functions overcomes them.

## Theoretical Background

### The Perceptron Learning Algorithm
A perceptron computes a weighted sum of its inputs and passes the result through
a step activation function, producing a binary output. Learning proceeds by the
weight update rule: `w(t+1) = w(t) + η(y − ŷ)x`, where `η` is the learning rate,
`y` is the true label, and `ŷ` is the predicted label. When a prediction is
correct, `(y − ŷ) = 0` and no update occurs; when incorrect, the weights are
nudged in the direction that would have made that example more likely to be
classified correctly.

The **Perceptron Convergence Theorem** guarantees that this algorithm will find a
separating hyperplane in a finite number of steps, but only if the data is
**linearly separable**. If no such hyperplane exists, the algorithm will never
converge — it will continue adjusting weights indefinitely without settling on a
stable solution. This is the fundamental limitation of a single-layer perceptron:
it can only represent linear decision boundaries, and it has no mechanism to
represent non-linear class structure, however it is trained.

### Extending a binary perceptron to multiple classes
A single perceptron inherently answers a binary (two-class) question. Multi-class
extension is achieved with a **One-vs-Rest** scheme: one binary perceptron is
trained per class, each learning to distinguish that class from all others. A
new point is classified by evaluating all perceptrons and selecting whichever
one produces the highest raw score, rather than relying on each one's individual
binary decision, since multiple perceptrons could otherwise disagree or agree
incorrectly.

### The Multilayer Perceptron
An MLP introduces one or more **hidden layers** between input and output, each
followed by a non-linear activation function (ReLU, sigmoid, or tanh). This
non-linearity is essential: stacking purely linear layers without non-linear
activations would collapse mathematically into a single linear transformation,
gaining nothing over a single-layer perceptron. With non-linear activations,
however, an MLP can approximate arbitrarily complex decision boundaries, a result
formalized by the **Universal Approximation Theorem**.

MLPs are trained via **backpropagation**: the network's prediction error is
propagated backward through the layers, computing how much each individual
weight contributed to the error (via the chain rule of calculus), and weights
are then adjusted via gradient descent (or a variant, such as Adam) to reduce
that error. The loss function for multi-class classification is typically
cross-entropy, which, analogous to log-loss in logistic regression, penalizes
confident incorrect predictions heavily.

### Key MLP hyperparameters and their theoretical role
The **activation function** determines what kinds of non-linear boundaries a
hidden layer can represent, and affects how gradients propagate during training
(for example, sigmoid and tanh can suffer from vanishing gradients in deep
networks, which ReLU largely avoids). The **optimizer** determines how weight
updates are computed from gradients — plain SGD applies a fixed-direction update
each step, while adaptive optimizers such as Adam adjust the effective step size
per parameter based on the history of gradients, generally converging faster and
more reliably. The **learning rate** controls the step size of each update: too
high risks overshooting and unstable training; too low risks painfully slow
convergence. Adding more hidden layers increases the network's representational
capacity, but does not automatically improve performance — beyond a certain
depth, without enough data or proper regularization, additional layers primarily
increase the risk of overfitting rather than improving generalization.

### Why PLA is expected to underperform MLP on complex data
Whenever the true decision boundary between classes is not linear — as is
typically the case for tasks like recognizing handwritten characters, where
classes overlap and vary in shape — a linear model is fundamentally incapable of
separating them well, no matter how it is trained or tuned. An MLP's hidden
layers and non-linear activations give it the representational flexibility to
bend its decision boundary to fit such structure, which is the theoretical basis
for expecting it to substantially outperform a single-layer perceptron on such
tasks.

## Dataset
A multi-class image classification dataset (handwritten characters) where class
boundaries are inherently non-linear, used to contrast a linear model's
representational limits against a non-linear one.
