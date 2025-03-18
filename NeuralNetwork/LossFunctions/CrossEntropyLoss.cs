using System;

namespace MNIST_NeuralNetwork.NeuralNetwork.LossFunctions
{
    public class CrossEntropyLoss : ILossFunction
    {
        // Compute the loss value
        public double Compute(double[] predictions, double[] targets)
        {
            double loss = 0;
            double[] softmaxPredictions = Softmax(predictions);

            for (int i = 0; i < predictions.Length; i++)
            {
                // Add a tiny offset (1e-9) to avoid log(0)
                loss += targets[i] * Math.Log(softmaxPredictions[i] + 1e-9);
            }
            return -loss;
        }

        // Compute the derivative of the loss w.r.t. predictions
        public double[] Derivative(double[] predictions, double[] targets)
        {
            double[] gradients = new double[predictions.Length];
            double [] softmaxPredictions = Softmax(predictions);
            //Console.WriteLine("Softmax Predictions: ");
            //for (int j = 0; j < softmaxPredictions.Length; j++)
            //{
            //    Console.Write($"{softmaxPredictions[j]:F3}, ");
            //}
            //Console.WriteLine();
            for (int i = 0; i < predictions.Length; i++)
            {
                // For (Softmax + CrossEntropy) scenario, derivative is (pred - target)
                gradients[i] = softmaxPredictions[i] - targets[i];
            }
            return gradients;
        }

        public double[] Softmax(double[] predictions)
        {
            double[] softmax = new double[predictions.Length];
            double sumExp = 0.0;
            for (int i = 0; i < predictions.Length; i++)
            {
                softmax[i] = Math.Exp(predictions[i]);
                sumExp += softmax[i];
            }

            for (int i = 0; i < predictions.Length; i++)
            {
                softmax[i] /= sumExp;
            }

            return softmax;
        }
    }
}