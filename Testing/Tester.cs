using MNIST_NeuralNetwork.Model;
using MNIST_NeuralNetwork.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_NeuralNetwork.Testing
{
    public class Tester
    {
        public static void TestModel(NeuralNetwork model, List<double[]> testInputs, List<double[]> testLabels)
        {
            model.setTrainingMode(false);
            double testLoss = Evaluator.EvaluateLoss(model, testInputs, testLabels);
            double testAcc = Evaluator.EvaluateAccuracy(model, testInputs, testLabels);
            Console.WriteLine($"Test Loss: {testLoss:F4}");
            Console.WriteLine($"Test Accuracy: {testAcc:P2}");
        }
    }
}
