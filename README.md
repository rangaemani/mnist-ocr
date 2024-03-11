# MNIST OCR From Scratch
### Description
Digit recognition OCR ML program built from scratch in Python w/ Pandas, no PyTorch, TensorFlow, etc.
### Structure
Breaks a 28 x 28 input image into rows and columns and passes it through the Layers of the Network. Standard neural network structure. Nodes comprise Layers, which combine to form a Network. Each Node has weights and biases to determine whether or not the signal is propogated through from one Layer to the next.
Generic Equation:|
Input Layer:: A0 = X (784 x m)
Some Propogated Layer Z:: Z1 = W1 * A0 + b1
Where W1 is some weight, and b1 is some bias factor.
### Propogation
This Network relies on Forward Propogation. Without any modification, each Node would be passed a piece of data that is simply a linear combination of all the input signals from each previous Node, plus some modification from the bias term, so in order to avoid that we will implement an Activation Function to it. This is what creates the "Hidden Layer" part of the neural network, allowing the SemNet to produce more useful results than just a series of linear combinations.
#### Activation Function
Using a pretty simple one known as RectLin (rectified linear) that is essentially a parametric equation described by the following:
RectLin = {
    if(x > 0){
        x
    } else if(x <= 0){
        0
    }
}

Which is basically what it sounds like, a truncated linear function. Seems deceivingly simple but this equation is what allows us to take advantage of the "Hidden Layer" model in order to create a "real" Neural Network.

So our equation for some A1 becomes the following:

A1 = g(Z1) = RectLin(Z1)

Which then gives us:

Z2 = W2 * A1 + b2

And then:

A2 = softmax(Z2)    
// Where softmax is defined as follows: "A function that converts a vector of K real numbers into a probability distribution of K possible outcomes"

This is what gives us the prediction from the neural network. The better the weights and biases, the more accurate the prediction.

### BackPropogation

This is how weights and biases are corrected for and the basic technique by which many models are trained. Essentially, you see where the model screwed up, and shift each weight and bias (using linear algebra) appropriately. The way this is calculated is very novel (to me at least) where you literally calculate error for the last layer, and then use that (along with the weights and biases) and literally calculate-in-reverse (backpropagate) the error through the function until you get to the start. Now that the function has been corrected, run it again. Rinse & repeat until your model is good.