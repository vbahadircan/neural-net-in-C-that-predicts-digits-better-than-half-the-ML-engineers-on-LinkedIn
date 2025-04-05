using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_NeuralNetwork.Model.Layers
{
    public class ActivationSoftmax : Layer
    {
        private double[] inputs;
        private double[] outputs;
        public override double[] Forward(double[] inputs)
        {
            this.inputs = inputs;
            outputs = new double[inputs.Length];

            // 1) Compute exponentials
            double maxVal = double.MinValue;
            for (int i = 0; i < inputs.Length; i++)
                if (inputs[i] > maxVal) maxVal = inputs[i];

            // Subtract maxVal for numerical stability
            double sumExp = 0.0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double expVal = Math.Exp(inputs[i] - maxVal);
                outputs[i] = expVal;
                sumExp += expVal;
            }

            // 2) Normalize
            for (int i = 0; i < outputs.Length; i++)
                outputs[i] /= sumExp;

            return outputs;
        }

        public override double[] Backward(double[] gradients, double learningRate)
        {
            throw new NotImplementedException();
        }
    }
}
