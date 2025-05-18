using MNIST_NeuralNetwork.Model;
using System;
using System.Collections.Generic;

namespace MNIST_NeuralNetwork.Utils
{
    public static class Evaluator
    {
        public static double EvaluateLoss(NeuralNetwork network, List<double[]> inputs, List<double[]> targets)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                double[] output = network.Forward(inputs[i]);
                totalLoss += network.lossFunction.Compute(output, targets[i]);
            }
            return totalLoss / inputs.Count;
        }

        public static double EvaluateAccuracy(NeuralNetwork network, List<double[]> inputs, List<double[]> targets)
        {
            int correct = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                double[] output = network.Forward(inputs[i]);

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

                // Compare with actual class (one-hot encoded target)
                int actualIndex = Array.IndexOf(targets[i], 1.0); // Find the index of the '1' in the one-hot encoded target
                if (predictedIndex == actualIndex)
                    correct++;
            }
            return (double)correct / inputs.Count;
        }
    }
}
