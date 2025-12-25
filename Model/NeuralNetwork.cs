using System;
using System.Collections.Generic;
using MNIST_NeuralNetwork.Model.Layers;
using MNIST_NeuralNetwork.Model.LossFunctions;
using MNIST_NeuralNetwork.Utils;

namespace MNIST_NeuralNetwork.Model
{
    public class NeuralNetwork
    {
        public List<Layer> layers = new List<Layer>();
        public ILossFunction lossFunction;
        public double learningRate;

        public NeuralNetwork(double learningRate, ILossFunction lossFunction)
        {
            this.learningRate = learningRate;
            this.lossFunction = lossFunction;
        }

        // Add a layer to the network
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
        }

        // Forward pass through all layers
        public double[] Forward(double[] input)
        {
            double[] output = input;
            foreach (var layer in layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public void setTrainingMode(bool isTraining)
        {
            foreach (var layer in layers)
            {
                if (layer is Dropout dropoutLayer)
                {
                    dropoutLayer.IsTraining = isTraining;
                }
            }
        }

        private class ModelDto
        {
            public List<LayerDto> Layers { get; set; }
        }

        private class LayerDto
        {
            public string Type { get; set; }
            public double[][] Weights { get; set; }

            public double[] Biases { get; set; }
            public double DropoutRate { get; set; }

        }

        public void Save(string path)
        {
            var modelDto = new ModelDto
            {
                Layers = new List<LayerDto>()
            };
            foreach (var layer in layers)
            {
                var layerDto = new LayerDto
                {
                    Type = layer.GetType().Name,
                };

                if (layer is DenseLayer denseLayer)
                {
                    layerDto.Weights = ArrayUtils.ToJagged(denseLayer.weights);
                    layerDto.Biases = denseLayer.biases;
                }
                else if (layer is ActivationReLU)
                {
                    // ActivationReLU has no weights or biases to save
                    layerDto.Weights = null;
                    layerDto.Biases = null;
                }

                else if (layer is Dropout dropoutLayer)
                {
                    layerDto.Weights = null;
                    layerDto.Biases = null;
                    layerDto.DropoutRate = dropoutLayer.dropoutRate;
                }

                modelDto.Layers.Add(layerDto);
            }
            string json = System.Text.Json.JsonSerializer.Serialize(modelDto);
            System.IO.File.WriteAllText(path, json);
        }

        public void Load(string path)
        {
            string json = System.IO.File.ReadAllText(path);
            var modelDto = System.Text.Json.JsonSerializer.Deserialize<ModelDto>(json);
            layers.Clear();
            foreach (var layerDto in modelDto.Layers)
            {
                Layer layer = null;
                if (layerDto.Type == nameof(DenseLayer))
                {
                    var denseLayer = new DenseLayer(layerDto.Weights[0].Length, layerDto.Weights.Length);
                    denseLayer.weights = ArrayUtils.To2D(layerDto.Weights);
                    denseLayer.biases = layerDto.Biases;
                    layer = denseLayer;
                }
                else if (layerDto.Type == nameof(ActivationReLU))
                {
                    layer = new ActivationReLU();
                }
                else if (layerDto.Type == nameof(Dropout))
                {
                    var dropoutLayer = new Dropout(layerDto.DropoutRate);
                    layer = dropoutLayer;
                }
                if (layer != null)
                {
                    layers.Add(layer);
                }
            }
        }
    }
}