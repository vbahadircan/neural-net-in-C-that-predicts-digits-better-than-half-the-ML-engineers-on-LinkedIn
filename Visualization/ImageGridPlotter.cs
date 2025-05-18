using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;

namespace MNIST_NeuralNetwork.Visualization
{
    public static class ImageGridPlotter
    {
        /// <summary>
        /// Renders a grid of digit images with predicted and actual labels below each.
        /// </summary>
        /// <param name="images">List of flattened 28×28 images (values 0–1).</param>
        /// <param name="predictions">Array of predicted digit labels.</param>
        /// <param name="actuals">Array of true digit labels.</param>
        /// <param name="gridCols">How many images per row in the grid.</param>
        /// <param name="scale">Pixel scaling factor for each MNIST pixel.</param>
        /// <param name="outputPath">Where to save the resulting PNG.</param>
        public static void SavePredictionGrid(
            List<double[]> images,
            int[] predictions,
            int[] actuals,
            int gridCols = 5,
            int scale = 10,
            string outputPath = "predictions.png")
        {
            if (images.Count != predictions.Length || images.Count != actuals.Length)
                throw new ArgumentException("Counts of images, predictions, and actuals must match.");

            const int imgSize = 28;                   // MNIST images are 28×28
            int count = images.Count;
            int gridRows = (int)Math.Ceiling(count / (double)gridCols);

            int cellWidth = imgSize * scale;
            int cellHeight = imgSize * scale + 20;    // extra space for the text

            int bmpWidth = gridCols * cellWidth;
            int bmpHeight = gridRows * cellHeight;

            using var bmp = new Bitmap(bmpWidth, bmpHeight);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.White);

            // Prepare drawing tools
            using var font = new Font(FontFamily.GenericSansSerif, 12);
            using var brush = new SolidBrush(Color.Black);

            for (int idx = 0; idx < count; idx++)
            {
                int row = idx / gridCols;
                int col = idx % gridCols;
                int offsetX = col * cellWidth;
                int offsetY = row * cellHeight;

                // 1) Draw the digit image
                var img = images[idx];
                for (int y = 0; y < imgSize; y++)
                {
                    for (int x = 0; x < imgSize; x++)
                    {
                        // MNIST: 0 = black, 1 = white (no inversion)
                        int gray = (int)(255 * img[y * imgSize + x]);
                        using var pixelBrush = new SolidBrush(Color.FromArgb(gray, gray, gray));
                        g.FillRectangle(
                            pixelBrush,
                            offsetX + x * scale,
                            offsetY + y * scale,
                            scale,
                            scale
                        );
                    }
                }

                // 2) Draw the predicted vs actual label
                string labelText = $"Predicted:{predictions[idx]}  Actual:{actuals[idx]}";
                var textSize = g.MeasureString(labelText, font);
                float textX = offsetX + (cellWidth - textSize.Width) / 2f;
                float textY = offsetY + imgSize * scale;
                g.DrawString(labelText, font, brush, textX, textY);
            }

            // 3) Save to file
            bmp.Save(outputPath, ImageFormat.Png);
            Console.WriteLine($"Prediction grid saved to {outputPath}");
        }
    }
}
