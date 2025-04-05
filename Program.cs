using System;
using System.Linq;
using MNIST_NeuralNetwork.Data;
using MNIST_NeuralNetwork.Model;
using MNIST_NeuralNetwork.Model.Layers;
using MNIST_NeuralNetwork.Model.LossFunctions;
using MNIST_NeuralNetwork.Training;
using MNIST_NeuralNetwork.Visualization;

namespace MNIST_NeuralNetwork
{
    
    class Program
{
    static void Main()
    {
        Console.WriteLine($"Current Directory: {Environment.CurrentDirectory}");

        string trainImagesPath = "Data/train/train-images.idx3-ubyte";
        string trainLabelsPath = "Data/train/train-labels.idx1-ubyte";

        string testImagesPath = "Data/test/t10k-images.idx3-ubyte";
        string testLabelsPath = "Data/test/t10k-labels.idx1-ubyte";

        // 1) Load Data
        var (trainImages, trainLabels) = MnistLoader.LoadImagesAndLabels(trainImagesPath, trainLabelsPath);
        List<double[]> trainLabelsEncoded = MnistLoader.OneHotEncoder(trainLabels);
        

        var (testImages, testLabels) = MnistLoader.LoadImagesAndLabels(testImagesPath, testLabelsPath);
        List<double[]> testLabelsEncoded = MnistLoader.OneHotEncoder(testLabels);

        // Split training data into training and validation sets
        double validationRatio = 0.2; // 20% for validation
        var (trainImagesSubset, valImages, trainLabelsEncodedSubset, valLabelsEncoded) = MnistLoader.SplitData(trainImages, trainLabelsEncoded, validationRatio);


        // Create a new neural network with learningRate=0.01 and CrossEntropyLoss
        NeuralNetwork model = new NeuralNetwork(learningRate: 0.01, new CrossEntropyLoss());

        // Build the network architecture
        model.AddLayer(new DenseLayer(784, 128));  // Hidden layer
        model.AddLayer(new ActivationReLU());    // Activation
        model.AddLayer(new DenseLayer(128, 128));   // Output layer
        model.AddLayer(new ActivationReLU()); // Activation
        model.AddLayer(new DenseLayer(128, 10));   // Output layer
        model.AddLayer(new ActivationReLU());


            // 2) Train the model
            Trainer trainer = new Trainer(model, 0.01);

            int totalEpochs = 10;
            trainer.TrainAll(trainImagesSubset, trainLabelsEncodedSubset, valImages, valLabelsEncoded, totalEpochs);

            // 3) Plot Results
            Plotter.PlotMetrics(trainer.epochList, trainer.trainLossList, trainer.valLossList, trainer.valAccList, trainer.epochTimeList);


            // Print a sample image data

            //double[] test = trainImages[0];
            //for (int i = 0; i < test.Length; i++)
            //{
            //    Console.Write(test[i] + " ");
            //    if (i % 28 == 0)
            //    {
            //        Console.WriteLine();
            //    }
            //}


            /*
            // Display trainLabels[0]
            Console.WriteLine(trainLabels[0]);

            // Display trainLabelsEncoded[0]
            for (int i = 0; i < trainLabelsEncoded[0].Length; i++)
            {
                Console.Write(trainLabelsEncoded[0][i] + " ");
            }
            */


        }
    }
}
