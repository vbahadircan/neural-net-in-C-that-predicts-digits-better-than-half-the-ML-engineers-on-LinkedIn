using System;

namespace MNIST_NeuralNetwork.Model.Layers
{
    public class ActivationReLU : Layer
    {
        private double[] inputs;

        public override double[] Forward(double[] inputs)
        {
            this.inputs = inputs;
            double[] outputs = new double[inputs.Length];

            // ReLU: max(0, x)
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = Math.Max(0, inputs[i]);
            }
            return outputs;
        }

        public override double[] Backward(double[] gradients, double learningRate)
        {
            double[] inputGradients = new double[inputs.Length];

            // ReLU derivative: 1 if input > 0, else 0
            for (int i = 0; i < inputs.Length; i++)
            {
                inputGradients[i] = inputs[i] > 0 ? gradients[i] : 0;
            }

            return inputGradients;
        }
    }
}