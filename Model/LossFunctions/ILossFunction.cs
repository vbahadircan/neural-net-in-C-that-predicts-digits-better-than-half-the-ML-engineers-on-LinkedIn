namespace MNIST_NeuralNetwork.Model.LossFunctions
{
    public interface ILossFunction
    {
        double Compute(double[] predictions, double[] targets);
        double[] Derivative(double[] predictions, double[] targets);
    }
}