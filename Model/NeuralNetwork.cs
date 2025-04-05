using System;
using System.Collections.Generic;
using MNIST_NeuralNetwork.Model.Layers;
using MNIST_NeuralNetwork.Model.LossFunctions;

namespace MNIST_NeuralNetwork.Model
{
    public class NeuralNetwork
    {
        public List<Layer> layers = new List<Layer>();
        public ILossFunction lossFunction;
        public double learningRate;

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
        
        public void TrainEpoch(List<double[]> inputs, List<double[]> targets, List<double[]> valInputs, List<double[]> valTargets,  int epochs)
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double totalLoss = 0.0;

                // Iterate through all samples in training set
                for (int i = 0; i < inputs.Count; i++)
                {
                    // 1) Forward
                    double[] output = Forward(inputs[i]);

                    // 2) Compute the loss (optional, for logging)
                    double lossValue = lossFunction.Compute(output, targets[i]);
                    totalLoss += lossValue;

                    // 3) Compute gradient of loss w.r.t. output
                    double[] lossGradient = lossFunction.Derivative(output, targets[i]);
                   

                    // 4) Backward pass
                    for (int layerIndex = layers.Count - 1; layerIndex >= 0; layerIndex--)
                    {
                        lossGradient = layers[layerIndex].Backward(lossGradient, learningRate);
                    }
                }
                double averageLoss = totalLoss / inputs.Count;
                double validationLoss = EvaluateValidationLoss(valInputs, valTargets);
                Console.WriteLine($"Epoch {epoch}/{epochs} -> Avg Loss: {averageLoss:F4}, Validation Loss: {validationLoss:F4}");
            }
        }

        public double EvaluateValidationLoss(List<double[]> valInputs, List<double[]> valTargets)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < valInputs.Count; i++)
            {
                double[] output = Forward(valInputs[i]);
                totalLoss += lossFunction.Compute(output, valTargets[i]);
            }
            return totalLoss / valInputs.Count;
        }

        // Test the model
        // This method takes targets as labels (not one-hot encoded)
        public double EvaluateAccuracy(List<double[]> inputs, List<int> targets)
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
                if (predictedIndex == actualIndex)
                    correct++;
            }
            return (double)correct / inputs.Count;
        }
    }
}