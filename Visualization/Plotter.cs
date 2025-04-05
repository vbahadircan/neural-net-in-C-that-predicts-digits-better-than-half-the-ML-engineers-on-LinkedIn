using System;
using System.Collections.Generic;
using ScottPlot;

namespace MNIST_NeuralNetwork.Visualization
{
    public static class Plotter
    {
        public static void PlotMetrics(
            List<int> epochs,
            List<double> trainLoss,
            List<double> valLoss,
            List<double> valAcc,
            List<double> epochTime,
            string saveFilePath = "metrics.png"
        )
        {
            // 1) Create a ScottPlot
            var plt = new ScottPlot.Plot(800, 600);

            // 2) Convert your Lists to arrays for easier plotting
            double[] xEpoch = epochs.ConvertAll(e => (double)e).ToArray();
            double[] yTrainLoss = trainLoss.ToArray();
            double[] yValLoss = valLoss.ToArray();
            double[] yValAcc = valAcc.ToArray();
            double[] yTime = epochTime.ToArray();

            // 3) Add the training/validation loss on the primary Y axis
            plt.AddScatter(xEpoch, yTrainLoss, label: "Train Loss", color: System.Drawing.Color.Blue);
            plt.AddScatter(xEpoch, yValLoss, label: "Val Loss", color: System.Drawing.Color.Red);

            // 4) Make a second Y axis for the accuracy if you want them on the same plot
            // Alternatively, you can create multiple subplots. Let's do the same plot with 2nd Y axis:
            var accAxis = plt.AddScatter(
                xEpoch, yValAcc,
                label: "Val Accuracy",
                color: System.Drawing.Color.Green
            );
            accAxis.YAxisIndex = 1;

            // 5) Optionally, adjust axis limits
            //   - Loss might go from near 0 to some max, so leave it auto
            //   - Accuracy is from 0 to 1
            plt.SetAxisLimits(yAxisIndex: 1);

            // 6) Label axes
            plt.YAxis.Label("Loss");
            plt.YAxis2.Label("Accuracy");
            plt.XAxis.Label("Epoch");

            // 7) Add Legend & Title
            plt.Legend();
            plt.Title("Training Metrics Over Epochs");

            // 8) Optionally plot epochTime in a separate chart or do another Y axis
            // For simplicity, let's do a separate subplot for time
            // We'll create a 2-row layout
            plt.Grid();
            // Additional approach would be subplots, shown below.

            // 9) Save or show figure
            plt.SaveFig(saveFilePath);
            Console.WriteLine($"Metrics plot saved to {saveFilePath}");
        }


        // Example approach if you want a second chart for epoch time
        public static void PlotEpochTime(
            List<int> epochs,
            List<double> epochTime,
            string saveFilePath = "epoch_time.png"
        )
        {
            var plt = new ScottPlot.Plot(800, 400);
            double[] xEpoch = epochs.ConvertAll(e => (double)e).ToArray();
            double[] yTime = epochTime.ToArray();

            plt.AddScatter(xEpoch, yTime, label: "Time per Epoch (sec)", color: System.Drawing.Color.Purple);
            plt.Title("Epoch Time");
            plt.XAxis.Label("Epoch");
            plt.YAxis.Label("Seconds");
            plt.Legend();
            plt.SaveFig(saveFilePath);

            Console.WriteLine($"EpochTime plot saved to {saveFilePath}");
        }
    }
}
