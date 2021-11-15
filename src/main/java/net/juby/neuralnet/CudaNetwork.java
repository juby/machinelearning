package net.juby.neuralnet;

import net.juby.neuralnet.mnist.MnistReader;
import org.apache.commons.lang3.time.StopWatch;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * A neural network that trains and identifies handwritten digits using the
 * <a href="http://yann.lecun.com/exdb/mnist/">MNIST database</a>. It utilizes
 * the <a href="http://www.jcuda.org/">JCuda</a> package to run some of the more
 * computationally demanding tasks on an NVIDIA graphics card using
 * <a href="https://developer.nvidia.com/cuda-toolkit>CUDA</a>.
 *
 * This first iteration will strictly use a quadratic cost function, additional
 * cost functions will be added later. Also, no command line options are supported
 * yet.
 *
 * @author Andrew Juby (jubydoo AT gmail DOT com)
 * @version 2.0 7/13/2021
 *
 */
public class CudaNetwork {
    private final int[] layerSizes;
    private final int numberOfLayers;
    private final float[][] biases;
    private final float[][][] weights;

    /**
     * Constructs a new CudaNetwork
     *
     * @param layerSizes an array containing the number of neurons in each layer
     */
    public CudaNetwork(int[] layerSizes) {
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        this.numberOfLayers = layerSizes.length;
        this.biases = new float[this.numberOfLayers - 1][];
        this.weights = new float[this.numberOfLayers - 1][][];
        Random rand = new Random(System.currentTimeMillis());

        // Initialize the weights and biases.

        // Create the vectors for each layer and initialize with random values.
        // biases[i] contains the biases for the (i+2)th layer.
        for(int i = 0; i < biases.length; i++){
            int vectorLength = layerSizes[i + 1];
            biases[i] = new float[vectorLength];
            for(int j = 0; j < vectorLength; j++){
                biases[i][j] = (float) rand.nextGaussian();
            }
        }
        // Create the weights matrices and initialize with random values.
        // weights[i] contains the weights connecting the (i+1)th layer to the
        // (i+2)th layer.
        for(int i = 0; i < weights.length; i++){
            int cols = layerSizes[i];
            int rows = layerSizes[i + 1];
            weights[i] = new float[rows][cols];
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    weights[i][j][k] = (float) rand.nextGaussian();
                }
            }
        }
    }

    public static void main(String[] args){
        // Training values setup.
        int[] values = new int[]{784, 30, 10};
        int epochs = 30;
        int miniBatchSize = 10;
        float eta = 3.0F;

        String dataFolder = "C:\\Users\\jubyd\\Projects\\machinelearning\\";
        String trainingLabelsFileLocation = dataFolder + "mnist_data\\train-labels.idx1-ubyte";
        String trainingDataFileLocation = dataFolder + "mnist_data\\train-images.idx3-ubyte";
        String testLabelsFileLocation = dataFolder + "mnist_data\\t10k-labels.idx1-ubyte";
        String testDataFileLocation = dataFolder + "mnist_data\\t10k-images.idx3-ubyte";

        // Variable initialization.
        int[] trainingLabels;
        int[] testLabels;
        List<int[][]> trainingData;
        List<int[][]> testData;
        StopWatch stopWatch = new StopWatch();
        CudaNetwork network = new CudaNetwork(values);
        float[][] trainingMatrix, testMatrix;

        // Extract the MNIST data.
        trainingLabels = MnistReader.getLabels(trainingLabelsFileLocation);
        trainingData = MnistReader.getImages(trainingDataFileLocation);
        testLabels = MnistReader.getLabels(testLabelsFileLocation);
        testData = MnistReader.getImages(testDataFileLocation);

        // Create the matrices to hold the converted data
        trainingMatrix = new float[trainingData.size()][trainingData.get(0).length * trainingData.get(0)[0].length];
        testMatrix = new float[testData.size()][testData.get(0).length * testData.get(0)[0].length];

        convertData(trainingData, testData, trainingMatrix, testMatrix);

        stopWatch.start();
        network.stochasticGradientDescent(trainingLabels, trainingMatrix,
                testLabels, testMatrix, epochs, miniBatchSize, eta);
        stopWatch.stop();
        System.out.println("Network trained and tested in " +
                stopWatch.getTime(TimeUnit.SECONDS) + " seconds.");
    }

    /**
     * Converts 2D int matrices into 1D arrays of floats normalized to be between
     * 0 and 1. Those arrays are then assembled into a single matrix each for
     * testing and training data, where each row represents one image.
     * @param trainingData training data extracted by MnistReader
     * @param testData testing data extracted by MnistReader
     * @param trainingMatrix collected matrix of all training examples
     * @param testMatrix collected matrix of all testing examples
     */
    private static void convertData(List<int[][]> trainingData, List<int[][]> testData,
                                    float[][] trainingMatrix, float[][] testMatrix){
        // Flatten training image data, normalize, and convert to float values.
        for (int i = 0; i < trainingData.size(); i++) {
            int[][] tempArray = trainingData.get(i);
            int rows = tempArray.length;
            int cols = tempArray[0].length;

            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    trainingMatrix[i][j * cols + k] = (float) (tempArray[j][k] / 255.0);
                }
            }
        }

        // Flatten test image data, normalize, and convert to float values.
        for (int l = 0; l < testData.size(); l++) {
            int[][] tempArray = testData.get(l);
            int rows = tempArray.length;
            int cols = tempArray[0].length;

            for (int m = 0; m < rows; m++) {
                for (int n = 0; n < cols; n++) {
                    testMatrix[l][m * cols + n] = (float) (tempArray[m][n] / 255.0);
                }
            }
        }
    }

    private float[] feedForward(float[] input){
        //TODO feedForward
        return input;
    }

    /**
     * Trains the neural network, then tests it.
     *
     * @param trainingLabels the correct result for each training example
     * @param trainingMatrix matrix of all training example data
     * @param testLabels the correct result for each testing example
     * @param testMatrix matrix of all testing example data
     * @param epochs number of times the network is to be trained and tested
     * @param miniBatchSize number of examples to train between updates
     * @param eta learning rate for the network
     */
    private void stochasticGradientDescent(int[] trainingLabels, float[][] trainingMatrix,
                                           int[] testLabels, float[][] testMatrix,
                                           int epochs, int miniBatchSize, float eta){
        // Local variable setup.
        int miniBatchCount = trainingMatrix.length/miniBatchSize;

        // Run all training examples for the specified number of epochs.
        for(int o = 0; o < epochs; o++){
            // Shuffle the examples.
            Collections.shuffle(Arrays.asList(trainingMatrix));

            for(int p = 0; p < miniBatchCount; p++){
                updateMiniBatch(Arrays.copyOfRange(trainingMatrix, p * miniBatchSize,
                        (p + 1) * miniBatchSize - 1), trainingLabels, eta);
            }

            //Output progress to command line.
            System.out.println("Epoch " + o + ": " + evaluate(testMatrix, testLabels) +
                    "/" + testMatrix.length);
        }
    }

    private int evaluate(float[][] testMatrix, int[] testLabels){
        //TODO: evaluate
        return 0;
    }

    private void updateMiniBatch(float[][] batch, int[] labels, float eta){
        // Create and initialize variables to hold changes for the biases and weights.
        float[][] nabla_b = new float[biases.length][];
        float[][] delta_nabla_b = new float[biases.length][];
        float[][][] nabla_w = new float[weights.length][][];
        float[][][] delta_nabla_w = new float[weights.length][][];
        for(int p = 0; p < biases.length; p++){
            nabla_b[p] = new float[biases[p].length];
            delta_nabla_b[p] = new float[biases[p].length];
            Arrays.fill(nabla_b[p], 0.0F);
        }
        for(int q = 0; q < weights.length; q++){
            nabla_w[q] = new float[weights[q].length][];
            delta_nabla_w[q] = new float[weights[q].length][];

            for(int r = 0; r < weights[q].length; r++){
                nabla_w[q][r] = new float[weights[q][r].length];
                delta_nabla_w[q][r] = new float[weights[q][r].length];
                Arrays.fill(nabla_w[q][r], 0.0F);
            }
        }

        // Run the backpropagation algorithm for each example in the batch
        for(int s = 0; s < batch.length; s++){
            // Reset the deltas
            for(int t = 0; t < biases.length; t++){
                Arrays.fill(delta_nabla_b[t], 0.0F);
            }
            for(int u = 0; u < weights.length; u++){
                for(int v = 0; v < weights[u].length; v++){
                    Arrays.fill(delta_nabla_w[u][v], 0.0F);
                }
            }

            backpropagation(delta_nabla_b, delta_nabla_w, batch[s], labels[s]);

            // TODO: Write CUDA code for updating the nabla_* matrices from the delta_nabla_* matrices
        }

        // TODO: Write CUDA code for updating weights and biases from the nabla_* matrices
    }

    private void backpropagation(float[][] delta_nabla_b,
                                 float[][][] delta_nabla_w,
                                 float[] trainingItem,
                                 int trainingLabel){
        //TODO: backpropagation
    }
}
