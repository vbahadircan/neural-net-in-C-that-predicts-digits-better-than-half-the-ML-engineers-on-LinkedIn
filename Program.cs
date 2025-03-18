using System;
using System.Linq;
using MNIST_NeuralNetwork.Data;
using MNIST_NeuralNetwork.NeuralNetwork;
using MNIST_NeuralNetwork.NeuralNetwork.Layers;
using MNIST_NeuralNetwork.NeuralNetwork.LossFunctions;

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
        int epochs = 10;

        Console.WriteLine("Training the neural network...");
        model.TrainEpoch(trainImages, trainLabelsEncoded, epochs);
        //model.TrainEpoch(testImages, testLabelsEncoded, epochs);

        //int epochs = 10;
        //int subsetSize = 1000; // Use a smaller subset of 100 samples

        //var trainImagesSubset = trainImages.Take(subsetSize).ToList();
        //var trainLabelsEncodedSubset = trainLabelsEncoded.Take(subsetSize).ToList();

        //Console.WriteLine("Training the neural network with a smaller subset...");
        //model.TrainEpoch(trainImagesSubset, trainLabelsEncodedSubset, epochs);



        //3) Evaluate the model
        Console.WriteLine("Evaluating the model...");
        double accuracy = model.Test(testImages, testLabels);
        Console.WriteLine($"Test accuracy: {accuracy:P2}");


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