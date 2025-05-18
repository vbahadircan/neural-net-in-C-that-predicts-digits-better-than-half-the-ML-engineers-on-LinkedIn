using System;

namespace MNIST_NeuralNetwork.Model.Layers
{
    public class Dropout : Layer
    {
        public double dropoutRate;
        private Random rand;
        private bool[] mask;
        private double[] inputs;

        public bool IsTraining { get; set; } = true;

       
        public Dropout(double rate)
        {
            dropoutRate = rate;
            rand = new Random();
        }

        public override double[] Forward(double[] input)
        {
            this.inputs = input;

            // If not training, just pass input as-is
            if (!IsTraining)
            {
                return input;
            }

            // Otherwise do normal dropout
            mask = new bool[input.Length];
            double[] outputs = new double[input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                bool keepNeuron = rand.NextDouble() > dropoutRate;
                mask[i] = keepNeuron;
                outputs[i] = keepNeuron ? input[i] : 0.0;
            }

            return outputs;
        }

        public override double[] Backward(double[] gradients, double learningRate)
        {
            // If not training, no neurons were dropped => pass gradients directly
            // But let's keep logic consistent: if not training, mask wasn't set. Just handle it.
            if (!IsTraining)
            {
                return gradients;
            }

            double[] gradForPrev = new double[gradients.Length];
            for (int i = 0; i < gradients.Length; i++)
            {
                gradForPrev[i] = mask[i] ? gradients[i] : 0.0;
            }
            return gradForPrev;
        }
    }
}
