using System;

namespace MNIST_NeuralNetwork.Model.Layers
{
    public class DenseLayer : Layer
    {
        private double[,] weights;
        private double[] biases;
        private double[] inputs;
        private double[] outputs;
        private Random rand = new Random();

        public int InputSize { get; }
        public int OutputSize { get; }

        public DenseLayer(int inputSize, int outputSize)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            weights = new double[outputSize, inputSize];
            biases = new double[outputSize];

            // Xavier Initialization for better training stability
            for (int i = 0; i < OutputSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                {
                    weights[i, j] = rand.NextDouble() * Math.Sqrt(1.0 / InputSize);
                }
            }
        }
            
        // Forward pass computes output of this layer
        public override double[] Forward(double[] inputs)
        {
            this.inputs = inputs;
            outputs = new double[OutputSize];

            for (int i = 0; i < OutputSize; i++)
            {
                double sum = biases[i];
                for (int j = 0; j < InputSize; j++)
                {
                    sum += weights[i, j] * inputs[j];
                }
                outputs[i] = sum;
            }
            return outputs;
        }

        // Backpropagation to update weights and compute gradients for previous layer
        public override double[] Backward(double[] gradients, double learningRate)
        {
            double[] inputGradients = new double[InputSize];

            for (int i = 0; i < OutputSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                {
                    // Accumulate gradient for the previous layer
                    inputGradients[j] += gradients[i] * weights[i, j];

                    // Gradient descent update for weights
                    weights[i, j] -= learningRate * gradients[i] * inputs[j];
                }
                // Update bias
                biases[i] -= learningRate * gradients[i];
            }

            return inputGradients;
        }
    }
}
