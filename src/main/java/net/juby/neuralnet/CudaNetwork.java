package net.juby.neuralnet;

import jcuda.*;
import java.util.Random;

/**
 * A neural network that trains and identifies handwritten digits using the
 * <a href="http://yann.lecun.com/exdb/mnist/">MNIST database</a>. It utilizes
 * the <a href="http://www.jcuda.org/">JCuda</a> package to run some of the more
 * computationally demanding tasks on an NVIDIA graphics card using
 * <a href="https://developer.nvidia.com/cuda-toolkit>CUDA</a>.
 *
 * This first iteration will strictly use a quadratic cost function, additional
 * cost functions will be added later.
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

    private double[] feedForward(double[] input){
        //TODO feedForward
        return input;
    }

    private void stochasticGradientDescent(double[][] trainingMatrix,
                                             double[][] testMatrix,
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
