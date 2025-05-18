using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_NeuralNetwork.Utils
{
    public static class ArrayUtils
    {
        // Convert a 2D array to a jagged array
        public static double[][] ToJagged(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var jagged = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                jagged[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                    jagged[i][j] = matrix[i, j];
            }
            return jagged;
        }

        // Convert a jagged array back to a 2D array
        public static double[,] To2D(double[][] jagged)
        {
            int rows = jagged.Length;
            int cols = jagged.Length > 0 ? jagged[0].Length : 0;
            var matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = jagged[i][j];
            return matrix;
        }
    }

}
