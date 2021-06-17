package net.juby;

import java.util.*;

import org.apache.commons.math3.linear.*;
import com.vsthost.rnd.commons.math.ext.linear.EMatrixUtils;
import net.juby.exceptions.MalformedInputDataException;
import net.juby.mnist.MnistReader;


public class Network {
    private final int[] layerSizes;
    private final int numberOfLayers;
    private final RealVector[] biases;
    private final RealMatrix[] weights;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        numberOfLayers = layerSizes.length;
        biases = new RealVector[numberOfLayers - 1];
        weights = new RealMatrix[numberOfLayers - 1];

        // Initialize the weights and biases.

        // Create the vectors for each layer and initialize with random values.
        for(int i = 1; i < biases.length; i++){
            int vectorLength = layerSizes[i];
            biases[i] = new ArrayRealVector(vectorLength);
            for(int j = 0; j < vectorLength; j++){
                biases[i].setEntry(j, Math.random());
            }
        }
        // Finally create the weights matrices and initialize with random values.
        for(int i = 0; i < weights.length; i++){
            int cols = layerSizes[i];
            int rows = layerSizes[i+1];
            weights[i] = new Array2DRowRealMatrix(rows, cols);
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    weights[i].setEntry(i, j, Math.random());
                }
            }
        }
    }

    /*
     * Run the program with a list of the number of neurons in each layer, for
     * example
     *      java Network 784 30 10
     * creates a neural network with 784 neurons in the input layer, 30 in a
     * single hidden layer, and 10 in the output layer.
     * @param args Command line arguments
     */
    public static void main(String[] args){
        // For now, I'm hardcoding these values. Down the line I'll rework the
        // main method to allow these values to be specified from the command
        // line.
        // todo Make command line arguments more flexible
        int epochs = 30;
        int miniBatchSize = 10;
        double eta = 3.0;

        // Extract the layer sizes from the command line.
        int[] values;
        try{
            values = Arrays.stream(args).mapToInt(Integer::parseInt).toArray();
        } catch (NumberFormatException e){
            throw new MalformedInputDataException("The list of neuron counts "+
                    "contains a value which is not a number.");
        }

        // Extract the MNIST data.
        int[] trainingLabels = MnistReader.getLabels("D:\\Documents\\Projects"+
                "\\machinelearning\\mnist_data\\train-labels.idx1-ubyte");
        List<int[][]> trainingData = MnistReader.getImages("D:\\Documents"+
                "\\Projects\\machinelearning\\mnist_data\\train-images.idx3-ubyte");

        int[] testLabels = MnistReader.getLabels("D:\\Documents\\Projects"+
                "\\machinelearning\\mnist_data\\t10k-labels.idx1-ubyte");
        List<int[][]> testData = MnistReader.getImages("D:\\Documents"+
                "\\Projects\\machinelearning\\mnist_data\\t10k-images.idx3-ubyte");

        // Generate the neural network.
        Network net = new Network(values);

        // Convert the data in to matrices we can use
        RealMatrix trainingMatrix, testMatrix;
        trainingMatrix = new Array2DRowRealMatrix(trainingData.size(),
                net.layerSizes[0] + 1);
        testMatrix =
                new Array2DRowRealMatrix(testData.size(), net.layerSizes[0] + 1);
        convertData(trainingData, trainingLabels, testData, testLabels, trainingMatrix, testMatrix);

        // Train the network using the MNIST data.
        net.stochasticGradientDescent(trainingMatrix, testMatrix,
                epochs, miniBatchSize, eta);
    }

    private static void convertData(List<int[][]> trainingData,
                             int[] trainingLabels,
                             List<int[][]> testData,
                             int[] testLabels,
                             RealMatrix trainingMatrix,
                             RealMatrix testMatrix){
        // Convert the training and test data from a List of 2D matrices into a
        // RealMatrix, where the first column is the label and the remaining columns
        // are the image data laid out such that each row begins in the position
        // following the last entry of the previous row. We do this so we can
        // easily shuffle the rows; later we can use utility methods to extract
        // submatrices in order to do the necessary linear algebra.
        for(int i = 0; i < trainingData.size(); i++){
            int[][] tempAry = trainingData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    double entryValue;

                    // I don't think that normalizing the entries to be between
                    // 0 and 1 (inclusive) is strictly necessary, but it doesn't
                    // hurt.
                    if(k == 0) entryValue = trainingLabels[k];
                    else entryValue = tempAry[j][k - 1]/255.0;

                    trainingMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }

        // Generate a RealMatrix of the test data.
        for(int i = 0; i < testData.size(); i++){
            int[][] tempAry = testData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;
            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    double entryValue;

                    if(k == 0) entryValue = testLabels[k];
                    else entryValue = tempAry[j][k - 1]/255.0;

                    testMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }
    }

    private RealVector feedForward(RealVector input){
        // The first layer is the input layer.
        RealVector ret = new ArrayRealVector(input);
        SigmoidVectorVisitor visitor = new SigmoidVectorVisitor();

        // For each layer, calculate a' = σ(wa+b).
        // [The operate() method multiplies the matrix by a given vector.]
        for(int i = 0; i < numberOfLayers; i++){
            ret = weights[i].operate(ret).add(biases[i]);
            ret.walkInOptimizedOrder(visitor);
        }

        // After all layers have been processed, ret contains the output layer.
        return ret;
    }

    private void stochasticGradientDescent(RealMatrix trainingMatrix,
                                          RealMatrix testMatrix,
                                          int epochs,
                                          int miniBatchSize,
                                          double eta){
        // Local variable setup.
        int nTest = testMatrix.getRowDimension();
        int miniBatchCount = trainingMatrix.getRowDimension()/miniBatchSize;
        RealMatrix[] miniBatches = new RealMatrix[miniBatchCount];

        // Run this loop for each epoch.
        for(int i = 0; i < epochs; i++) {
            //Randomize the training data.
            trainingMatrix = EMatrixUtils.shuffleRows(trainingMatrix);

            //Generate the mini batches.
            for (int j = 0; j < miniBatchCount; j++) {
                int start = j * miniBatchCount;
                miniBatches[j] = trainingMatrix.getSubMatrix(
                        start,
                        start + miniBatchCount - 1,
                        0,
                        trainingMatrix.getColumnDimension() - 1
                );
            }

            // Run the mini batches.
            for (RealMatrix batch : miniBatches) {
                updateMiniBatch(batch, eta);
            }

            //Output progress to command line.
            System.out.println("Epoch " + i + ": " + evaluate(testMatrix) +
                    "/" + nTest);

        }
    }

    // Runs test data through the network and identifies the number of correct
    // answers, 'correct' being that the neuron corresponding to the desired
    // result has the highest activation
    private int evaluate(RealMatrix testData) {
        int total, targetValue, resultValue;
        total = 0;

        // Loop through all of the test data
        for(int i = 0; i < testData.getRowDimension(); i++){
            targetValue = (int) testData.getEntry(i, 0);

            // Extract the test case from the matrix, feed it through, and get
            // the max index (as neuron 0 represents the neuron with the net's
            // 'guess' at how likely it's a 0, and so on).
            resultValue = feedForward(
                    new ArrayRealVector(testData.getRow(i), 1,
                            testData.getColumnDimension() - 1)
                ).getMaxIndex();

            // Add to the running tally if it's a correct answer.
            if(targetValue == resultValue) total += 1;
        }
        return total;
    }

    // From the sample code in the textbook:
    // Update the network's weights and biases by applying gradient descent
    // using backpropagation to a single mini batch. The "batch" is a list of
    // tuples "(x, y)", and "eta" is the learning rate.
    private void updateMiniBatch(RealMatrix batch, double eta) {
        // These are the vectors and matrices that will store the changes to the
        // weights and biases during each epoch.
        RealVector[] nabla_b = new RealVector[biases.length];
        RealVector[] delta_nabla_b = new RealVector[biases.length];
        RealMatrix[] nabla_w = new RealMatrix[weights.length];
        RealMatrix[] delta_nabla_w = new RealMatrix[weights.length];

        // Set up the nabla arrays
        for(int r = 0; r < biases.length; r++){
            nabla_b[r] =
                    new ArrayRealVector(biases[r].getDimension(), 0.0);
            delta_nabla_b[r] =
                    new ArrayRealVector(biases[r].getDimension(), 0.0);
        }
        for(int s = 0; s < weights.length; s++){
            int rows = weights[s].getRowDimension();
            int cols = weights[s].getColumnDimension();

            nabla_w[s] = new Array2DRowRealMatrix(rows, cols);
            delta_nabla_w[s] = new Array2DRowRealMatrix(rows, cols);

            for(int t = 0; t < cols; t++){
                nabla_w[s].setColumnVector(t, new ArrayRealVector(rows, 0.0));
                delta_nabla_w[s].setColumnVector(t, new ArrayRealVector(rows, 0.0));
            }
        }

        //Run the backpropagation algorithm for each entry in the batch.
        for(int i = 0; i < batch.getRowDimension(); i++){
            backpropagation(delta_nabla_b, delta_nabla_w, batch.getRowVector(i));
            for(int j = 0; j < nabla_b.length; j++){
                nabla_b[j].add(delta_nabla_b[j]);
            }
            for(int k = 0; k < nabla_w.length; k++){
                nabla_w[k].add(delta_nabla_w[k]);
            }
        }

        //Update the weights matrices.
        for(int l = 0; l < weights.length; l++){
            for(int m = 0; m < weights[l].getRowDimension(); m++){
                for(int n = 0; n < weights[l].getColumnDimension(); n++){
                    double current = weights[l].getEntry(m, n);
                    double delta = (eta/batch.getRowDimension())*nabla_w[l].getEntry(m, n);
                    weights[l].setEntry(m, n, current - delta);
                }
            }
        }

        //Update the bias vectors.
        for(int p = 0; p < biases.length; p++){
            for(int q = 0; q < biases[p].getDimension(); q++){
                double current = biases[p].getEntry(q);
                double delta = (eta/batch.getRowDimension()) * nabla_b[p].getEntry(q);
                biases[p].setEntry(q, current - delta);
            }
        }
    }

    private void backpropagation(RealVector[] delta_nabla_b,
                                 RealMatrix[] delta_nabla_w,
                                 RealVector trainingItem) {
        // First, let's extract the label from the rest of the data and put it
        // into a vector for when we need to compute the cost derivative.
        RealVector desiredActivations = new ArrayRealVector(10, 0.0);
        desiredActivations.setEntry((int) trainingItem.getEntry(0), 1.0);

        // Reset the deltas
        // Set up the nabla arrays
        for(int r = 0; r < biases.length; r++){
            delta_nabla_b[r] =
                    new ArrayRealVector(biases[r].getDimension(), 0.0);
        }
        for(int s = 0; s < weights.length; s++){
            int rows = weights[s].getRowDimension();
            int cols = weights[s].getColumnDimension();

            delta_nabla_w[s] = new Array2DRowRealMatrix(rows, cols);

            for(int t = 0; t < cols; t++){
                delta_nabla_w[s].setColumnVector(t, new ArrayRealVector(rows, 0.0));
            }
        }

        // Local variable setup
        RealVector[] activations = new RealVector[this.numberOfLayers];
        activations[0] = trainingItem.getSubVector(1, trainingItem.getDimension() - 1);
        RealVector[] weightedInputs = new RealVector[this.numberOfLayers - 1];
        SigmoidVectorVisitor sigmoidVectorVisitor = new SigmoidVectorVisitor();
        SigmoidPrimeVectorVisitor sigmoidPrimeVectorVisitor = new SigmoidPrimeVectorVisitor();
        RealVector z;

        // Feed forward
        for(int i = 0; i < this.numberOfLayers - 1; i++){
            weightedInputs[i] = weights[i].operate(activations[i]).add(biases[i]);
            activations[i + 1] = weightedInputs[i];
            activations[i + 1].walkInOptimizedOrder(sigmoidVectorVisitor);
        }

        // Backward pass
        RealVector sigmoidPrimeWL = weightedInputs[weightedInputs.length - 1];
        sigmoidPrimeWL.walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
        RealVector delta =
                costDerivative(activations[this.numberOfLayers - 1], desiredActivations)
                        .ebeMultiply(sigmoidPrimeWL);
        delta_nabla_b[delta_nabla_b.length - 1] = delta.copy();
        delta_nabla_w[delta_nabla_w.length - 1] =
                delta.outerProduct(activations[this.numberOfLayers - 2]);

        for(int l = this.numberOfLayers - 2; l >= 0; l--){
            z = weightedInputs[l].copy();
            z.walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
            delta = this.weights[l + 1].transpose().operate(delta).ebeMultiply(z);
            delta_nabla_b[l] = delta.copy();
            delta_nabla_w[l] = delta.outerProduct(activations[l-1]);
        }
    }

    private static RealVector costDerivative(RealVector outputActivations, RealVector desiredActivations){
        return outputActivations.subtract(desiredActivations);
    }
}
