namespace MNIST_NeuralNetwork.NeuralNetwork.LossFunctions
{
    public interface ILossFunction
    {
        double Compute(double[] predictions, double[] targets);
        double[] Derivative(double[] predictions, double[] targets);
    }
}