using System;
using System.Collections.Generic;
using System.Diagnostics;
using MNIST_NeuralNetwork.Model;
using MNIST_NeuralNetwork.Model.Layers;
using MNIST_NeuralNetwork.Model.LossFunctions;

namespace MNIST_NeuralNetwork.Training
{
    public class Trainer
    {
        public NeuralNetwork network;
        private double learningRate;
        public List<int> epochList = new List<int>();
        public List<double> trainLossList = new List<double>();
        public List<double> valLossList = new List<double>();
        public List<double> valAccList = new List<double>();
        public List<double> epochTimeList = new List<double>();

        public Trainer(NeuralNetwork net, double lr)
        {
            network = net;
            learningRate = lr;
        }

        // Train for multiple epochs
        public void TrainAll(
            List<double[]> trainInputs, List<double[]> trainLabels,
            List<double[]> valInputs, List<double[]> valLabels,
            int totalEpochs)
        {
            for (int epoch = 1; epoch <= totalEpochs; epoch++)
            {
                // 1) Time the training for this epoch
                Stopwatch sw = Stopwatch.StartNew();
                double trainLoss = TrainOneEpoch(trainInputs, trainLabels);
                sw.Stop();
                double elapsedSeconds = sw.Elapsed.TotalSeconds;

                // 2) Evaluate on validation
                double valLoss = EvaluateLoss(valInputs, valLabels);
                double valAcc = EvaluateAccuracy(valInputs, valLabels);

                // 3) Store metrics
                epochList.Add(epoch);
                trainLossList.Add(trainLoss);
                valLossList.Add(valLoss);
                valAccList.Add(valAcc);
                epochTimeList.Add(elapsedSeconds);

                Console.WriteLine($"Epoch {epoch}/{totalEpochs} => "
                    + $"TrainLoss={trainLoss:F4}, ValLoss={valLoss:F4}, ValAcc={valAcc:P2}, Time={elapsedSeconds:F2}s");
            }
        }

        // Return average train loss for the epoch
        private double TrainOneEpoch(List<double[]> trainInputs, List<double[]> trainLabels)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < trainInputs.Count; i++)
            {
                double[] output = network.Forward(trainInputs[i]);
                double sampleLoss = network.lossFunction.Compute(output, trainLabels[i]);
                totalLoss += sampleLoss;

                // Backprop
                double[] grad = network.lossFunction.Derivative(output, trainLabels[i]);
                for (int layerIndex = network.layers.Count - 1; layerIndex >= 0; layerIndex--)
                    grad = network.layers[layerIndex].Backward(grad, learningRate);
            }
            return totalLoss / trainInputs.Count;
        }

        private double EvaluateLoss(List<double[]> inputs, List<double[]> labels)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                var output = network.Forward(inputs[i]);
                totalLoss += network.lossFunction.Compute(output, labels[i]);
            }
            return totalLoss / inputs.Count;
        }

        private double EvaluateAccuracy(List<double[]> inputs, List<double[]> labels)
        {
            int correct = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                double[] output = network.Forward(inputs[i]);
                int predicted = Array.IndexOf(output, output.Max());
                int actual = Array.IndexOf(labels[i], 1);
                if (predicted == actual)
                    correct++;
            }
            return (double)correct / inputs.Count;
        }

        // Expose these lists so Plotter can use them
        public List<int> Epochs => epochList;
        public List<double> TrainLoss => trainLossList;
        public List<double> ValLoss => valLossList;
        public List<double> ValAcc => valAccList;
        public List<double> EpochTime => epochTimeList;
    }
}
