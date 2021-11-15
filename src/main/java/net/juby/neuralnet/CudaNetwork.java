package net.juby.neuralnet;

import jcuda.*;
import net.juby.neuralnet.mnist.MnistReader;
import org.apache.commons.lang3.time.StopWatch;

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
    private final double[][] biases;
    private final double[][][] weights;

    /**
     * Constructs a new CudaNetwork
     *
     * @param layerSizes an array containing the number of neurons in each layer
     */
    public CudaNetwork(int[] layerSizes) {
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        this.numberOfLayers = layerSizes.length;
        this.biases = new double[this.numberOfLayers - 1][];
        this.weights = new double[this.numberOfLayers - 1][][];
        Random rand = new Random(System.currentTimeMillis());

        // Initialize the weights and biases.

        // Create the vectors for each layer and initialize with random values.
        // biases[i] contains the biases for the (i+2)th layer.
        for(int i = 0; i < biases.length; i++){
            int vectorLength = layerSizes[i + 1];
            biases[i] = new double[vectorLength];
            for(int j = 0; j < vectorLength; j++){
                biases[i][j] = rand.nextGaussian();
            }
        }
        // Create the weights matrices and initialize with random values.
        // weights[i] contains the weights connecting the (i+1)th layer to the
        // (i+2)th layer.
        for(int i = 0; i < weights.length; i++){
            int cols = layerSizes[i];
            int rows = layerSizes[i + 1];
            weights[i] = new double[rows][cols];
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    weights[i][j][k] = rand.nextGaussian();
                }
            }
        }
    }

    public static void main(String[] args){
        // Training values setup.
        int[] values = new int[]{784, 30, 10};
        int epochs = 30;
        int miniBatchSize = 10;
        double eta = 3.0;

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
        double[][] trainingMatrix = null, testMatrix = null;

        // Extract the MNIST data.
        trainingLabels = MnistReader.getLabels(trainingLabelsFileLocation);
        trainingData = MnistReader.getImages(trainingDataFileLocation);
        testLabels = MnistReader.getLabels(testLabelsFileLocation);
        testData = MnistReader.getImages(testDataFileLocation);

        convertData(trainingData, testData, trainingMatrix, testMatrix);

        stopWatch.start();
        network.stochasticGradientDescent(trainingLabels, trainingMatrix,
                testLabels, testMatrix, epochs, miniBatchSize, eta);
        stopWatch.stop();
        System.out.println("Network trained and tested in " +
                stopWatch.getTime(TimeUnit.SECONDS) + " seconds.");
    }

    private static void convertData(List<int[][]> trainingData, List<int[][]> testData,
                                    double[][] trainingMatrix, double[][] testMatrix){
        trainingMatrix = new double[trainingData.get(0).length][trainingData.get(0)[0].length];
        testMatrix = new double[testData.get(0).length][testData.get(0)[0].length];

        // Flatten training image data, normalize, and convert to double values.
        for (int[][] tempArray : trainingData) {
            for (int j = 0; j < tempArray.length; j++) {
                for (int k = 0; k < tempArray[j].length; k++) {
                    trainingMatrix[j][k] = tempArray[j][k] / 255.0;
                }
            }
        }

        // Flatten test image data, normalize, and convert to double values.
        for (int[][] tempArray : testData) {
            for (int m = 0; m < tempArray.length; m++) {
                for (int n = 0; n < tempArray[m].length; n++) {
                    testMatrix[m][n] = tempArray[m][n] / 255.0;
                }
            }
        }
    }

    private double[] feedForward(double[] input){
        //TODO feedForward
        return input;
    }

    private void stochasticGradientDescent(int[] trainingLabels, double[][] trainingMatrix,
                                           int[] testLabels, double[][] testMatrix,
                                           int epochs,
                                           int miniBatchSize,
                                           double eta){
        //TODO: SGD
    }

    private int evaluate(double[][] testData){
        //TODO: evaluate
        return 0;
    }

    private void updateMiniBatch(double[][] batch, double eta){
        //TODO: updateMiniBatch
    }

    private void backpropagation(double[][] delta_nabla_b,
                                 double[][][] delta_nabla_w,
                                 double[] trainingItem){
        //TODO: backpropagation
    }
}
