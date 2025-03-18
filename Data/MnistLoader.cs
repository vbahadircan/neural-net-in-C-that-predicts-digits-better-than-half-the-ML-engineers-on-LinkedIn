using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace MNIST_NeuralNetwork.Data
{
    public static class MnistLoader
    {
        private const int ImagesMagicNumber = 2051;
        private const int LabelsMagicNumber = 2049;

        public static (List<double[]>, List<int>) LoadImagesAndLabels(string imagesPath, string labelsPath)
        {
            ValidateFileExists(imagesPath, "Images file");
            ValidateFileExists(labelsPath, "Labels file");

            try
            {
                var images = ReadImageFile(imagesPath);
                var labels = ReadLabelFile(labelsPath);

                if (images.Count != labels.Count)
                {
                    throw new InvalidDataException($"Mismatched counts: {images.Count} images but {labels.Count} labels");
                }

                return (images, labels);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new IOException($"Error accessing MNIST data files: {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error processing MNIST data: {ex.Message}", ex);
            }
        }

        // Async version for better performance with large datasets
        public static async Task<(List<double[]>, List<int>)> LoadImagesAndLabelsAsync(string imagesPath, string labelsPath)
        {
            ValidateFileExists(imagesPath, "Images file");
            ValidateFileExists(labelsPath, "Labels file");

            try
            {
                // Run both file reads concurrently for better performance
                var imagesTask = Task.Run(() => ReadImageFile(imagesPath));
                var labelsTask = Task.Run(() => ReadLabelFile(labelsPath));

                await Task.WhenAll(imagesTask, labelsTask);

                var images = imagesTask.Result;
                var labels = labelsTask.Result;

                if (images.Count != labels.Count)
                {
                    throw new InvalidDataException($"Mismatched counts: {images.Count} images but {labels.Count} labels");
                }

                return (images, labels);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new IOException($"Error accessing MNIST data files: {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error processing MNIST data: {ex.Message}", ex);
            }
        }

        private static List<double[]> ReadImageFile(string imagesPath)
        {
            using var fileStream = new FileStream(imagesPath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream);

            // Validate magic number
            int magic = ReverseInt32(reader.ReadInt32());
            if (magic != ImagesMagicNumber)
                throw new InvalidDataException($"Invalid magic number in images file: expected {ImagesMagicNumber}, got {magic}");

            // Read header info
            int numImages = ReverseInt32(reader.ReadInt32());
            int numRows = ReverseInt32(reader.ReadInt32());
            int numCols = ReverseInt32(reader.ReadInt32());
            int pixelsPerImage = numRows * numCols;

            // Read image data
            var images = new List<double[]>(numImages);
            for (int i = 0; i < numImages; i++)
            {
                var imageData = new double[pixelsPerImage];

                // Read all bytes for this image at once for better performance
                byte[] pixelBytes = reader.ReadBytes(pixelsPerImage);
                for (int j = 0; j < pixelsPerImage; j++)
                {
                    // Normalize to [0, 1]
                    imageData[j] = pixelBytes[j] / 255.0;
                }

                images.Add(imageData);
            }

            return images;
        }

        private static List<int> ReadLabelFile(string labelsPath)
        {
            using var fileStream = new FileStream(labelsPath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream);

            // Validate magic number
            int magic = ReverseInt32(reader.ReadInt32());
            if (magic != LabelsMagicNumber)
                throw new InvalidDataException($"Invalid magic number in labels file: expected {LabelsMagicNumber}, got {magic}");

            // Read number of labels
            int numLabels = ReverseInt32(reader.ReadInt32());

            // Read label data
            var labels = new List<int>(numLabels);
            for (int i = 0; i < numLabels; i++)
            {
                labels.Add(reader.ReadByte());
            }

            return labels;
        }

        public static List<double[]> OneHotEncoder(List<int> labels, int numClasses = 10)
        {
            var oneHotLabels = new List<double[]>(labels.Count);

            foreach (int label in labels)
            {
                if (label < 0 || label >= numClasses)
                {
                    throw new ArgumentOutOfRangeException(nameof(labels),
                        $"Label {label} is outside the valid range [0-{numClasses - 1}]");
                }

                var oneHotLabel = new double[numClasses];
                oneHotLabel[label] = 1.0;
                oneHotLabels.Add(oneHotLabel);
            }

            return oneHotLabels;
        }

        // Helper: Convert from big-endian to little-endian
        private static int ReverseInt32(int value)
        {
            return BitConverter.IsLittleEndian
                ? ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | ((value & 0xFF0000) >> 8) | ((value >> 24) & 0xFF)
                : value;
        }

        private static void ValidateFileExists(string filePath, string fileDescription)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException($"{fileDescription} path cannot be null or empty", nameof(filePath));
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"{fileDescription} not found at path: {filePath}", filePath);
            }
        }
    }
}