using System;


namespace MNIST_NeuralNetwork.NeuralNetwork.Layers
{
    public abstract class Layer
    {
        // Forward pass: Takes inputs, produces outputs
        public abstract double[] Forward(double[] inputs);

        // Backward pass: Takes gradients from next layer, returns gradients for previous layer
        public abstract double[] Backward(double[] gradients, double learningRate);
    }
}
