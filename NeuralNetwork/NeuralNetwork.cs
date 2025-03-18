using System;
using System.Collections.Generic;
using MNIST_NeuralNetwork.NeuralNetwork.Layers;
using MNIST_NeuralNetwork.NeuralNetwork.LossFunctions;

namespace MNIST_NeuralNetwork.NeuralNetwork
{
    public class NeuralNetwork
    {
        private List<Layer> layers = new List<Layer>();
        private ILossFunction lossFunction;
        private double learningRate;

        public NeuralNetwork(double learningRate, ILossFunction lossFunction)
        {
            this.learningRate = learningRate;
            this.lossFunction = lossFunction;
        }

        // Add a layer to the network
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
        }

        // Forward pass through all layers
        public double[] Forward(double[] input)
        {
            double[] output = input;
            foreach (var layer in layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        // Train one step using backpropagation
        
        public void Train(double[] input, double[] target)
        {
            // 1) Forward
            double[] output = Forward(input);

            // 2) Compute the gradient of loss w.r.t. output
            double[] lossGradient = lossFunction.Derivative(output, target);

            // 3) Backward
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                lossGradient = layers[i].Backward(lossGradient, learningRate);
            }
        }
        
        public void TrainEpoch(List<double[]> inputs, List<double[]> targets, int epochs)
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double totalLoss = 0.0;

                // Iterate through all samples in training set
                for (int i = 0; i < inputs.Count; i++)
                {
                    // 1) Forward
                    double[] output = Forward(inputs[i]);
                    //Console.WriteLine("Output: ");
                    //for (int j = 0; j < output.Length; j++)
                    //{
                    //    Console.Write($"{output[j]:F4}, ");
                    //}

                    // 2) Compute the loss (optional, for logging)
                    double lossValue = lossFunction.Compute(output, targets[i]);
                    //Console.WriteLine($"Loss: {lossValue:F4}"); 
                    totalLoss += lossValue;

                    // 3) Compute gradient of loss w.r.t. output
                    double[] lossGradient = lossFunction.Derivative(output, targets[i]);
                    //Console.WriteLine("Loss Gradient: ");
                    //for (int j = 0; j < lossGradient.Length; j++)
                    //{
                    //    Console.Write($"{lossGradient[j]:F4}, ");
                    //}

                    // 4) Backward pass
                    for (int layerIndex = layers.Count - 1; layerIndex >= 0; layerIndex--)
                    {
                        lossGradient = layers[layerIndex].Backward(lossGradient, learningRate);
                    }
                    //Console.WriteLine();
                }

                // Optionally print epoch info
                double averageLoss = totalLoss / inputs.Count;
                Console.WriteLine($"Epoch {epoch}/{epochs} -> Avg Loss: {averageLoss:F4}");
            }
        }

        // Test the model
        // This method takes targets as labels (not one-hot encoded)
        public double Test(List<double[]> inputs, List<int> targets)
        {
            int correct = 0; 
            for (int i = 0; i < inputs.Count; i++)
            {
                // Forward pass
                double[] output = Forward(inputs[i]);

                // Find predicted class
                int predictedIndex = 0;
                double maxVal = double.MinValue;
                for (int j = 0; j < output.Length; j++)
                {
                    if (output[j] > maxVal)
                    {
                        maxVal = output[j];
                        predictedIndex = j;
                    }
                }

                // Find actual class
                int actualIndex = (int)targets[i];
                //Console.WriteLine($"Predicted: {predictedIndex}, Actual: {actualIndex}");
                // Compare
                if (predictedIndex == actualIndex)
                    correct++;
            }

            // Return accuracy
            return (double)correct / inputs.Count;
        }
    }
}