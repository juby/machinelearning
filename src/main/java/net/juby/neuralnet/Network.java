package net.juby.neuralnet;

import java.util.*;

import net.juby.costFunctions.*;
import net.juby.visitors.*;
import org.apache.commons.math3.linear.*;
import com.vsthost.rnd.commons.math.ext.linear.EMatrixUtils;
import net.juby.exceptions.*;
import net.juby.mnist.*;

/**
 * A neural network that trains and identifies handwritten digits using the
 * <a href="http://yann.lecun.com/exdb/mnist/">MNIST database</a>.
 *
 * @author Andrew Juby (jubydoo AT gmail DOT com)
 * @version 1.0.2, 07/09/2021
 *
 */
public class Network {
    private final int[] layerSizes;
    private final int numberOfLayers;
    private final RealVector[] biases;
    private final RealMatrix[] weights;
    private final CostFunction costFunction;

    /**
     * Initializes a new neural network.
     * @param layerSizes array of the number of neurons in each layer
     */
    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        this.numberOfLayers = layerSizes.length;
        this.biases = new RealVector[numberOfLayers - 1];
        this.weights = new RealMatrix[numberOfLayers - 1];
        this.costFunction = new QuadraticCost();
        Random rand = new Random(System.currentTimeMillis());

        // Initialize the weights and biases.

        // Create the vectors for each layer and initialize with random values.
        // biases[i] contains the biases for the (i+2)th layer.
        for(int i = 0; i < biases.length; i++){
            int vectorLength = layerSizes[i + 1];
            biases[i] = new ArrayRealVector(vectorLength);
            for(int j = 0; j < vectorLength; j++){
                biases[i].setEntry(j, rand.nextGaussian());
            }
        }
        // Finally create the weights matrices and initialize with random values.
        // weights[i] contains the weights connecting the (i+1)th layer to the
        // (i+2)th layer.
        for(int i = 0; i < weights.length; i++){
            int cols = layerSizes[i];
            int rows = layerSizes[i + 1];
            weights[i] = new BlockRealMatrix(rows, cols);
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    weights[i].setEntry(j, k, rand.nextGaussian());
                }
            }
        }
    }

    /**
     * Runs the program with a list of the number of neurons in each layer, for
     * example
     * <code>java Network 784 30 10</code>
     * creates a neural network with 784 neurons in the input layer, 30 in a
     * single hidden layer, and 10 in the output layer.
     * @param args list of neurons in each layer
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
        trainingMatrix = new BlockRealMatrix(trainingData.size(),
                net.layerSizes[0] + 1);
        testMatrix =
                new BlockRealMatrix(testData.size(), net.layerSizes[0] + 1);
        convertData(trainingData, trainingLabels, testData, testLabels, trainingMatrix, testMatrix);

        // Train the network using the MNIST data.
        net.stochasticGradientDescent(trainingMatrix, testMatrix,
                epochs, miniBatchSize, eta);
    }

    /**
     * Converts the data as read by {@link MnistReader} into {@link RealMatrix}
     * objects. Each row in the resulting matrix contains data for one image,
     * where the first position contains the correct identification for that image
     * and the remainder are the greyscale pixel values, with the sequential
     * rows one after another.
     *
     * @param trainingData a List of integer arrays, each array contains the grayscale values for each pixel in a training image
     * @param trainingLabels the correct identification for each training example
     * @param testData a List of integer arrays, each array contains the grayscale values for each pixel in a testing image
     * @param testLabels the correct identification for each testing example
     * @param trainingMatrix the RealMatrix that holds the converted training data
     * @param testMatrix the RealMatrix that holds the converted testing data
     */
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

            trainingMatrix.setEntry(i, 0, trainingLabels[i]);

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols; k++){
                    trainingMatrix.setEntry(i, j*tempAryCols + k + 1, tempAry[j][k]/255.0);
                }
            }
        }

        // Generate a RealMatrix of the test data.
        for(int i = 0; i < testData.size(); i++){
            int[][] tempAry = testData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;

            testMatrix.setEntry(i, 0, testLabels[i]);

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols; k++){
                    testMatrix.setEntry(i, j*tempAryCols + k + 1, tempAry[j][k]/255.0);
                }
            }
        }
    }

    /**
     * Takes a {@link RealVector} as input to the network and processes it.
     *
     * @param input the RealVector that has the values for the input neurons.
     * @return a value corresponding to the neuron with the highest activation on the output layer
     */
    private RealVector feedForward(RealVector input){
        // The first layer is the input layer.
        RealVector ret = new ArrayRealVector(input);
        SigmoidVectorVisitor visitor = new SigmoidVectorVisitor();

        // For each layer, calculate a' = σ(wa+b).
        // [The operate() method multiplies the matrix by a given vector.]
        for(int i = 1; i < numberOfLayers; i++){
            ret = weights[i - 1].operate(ret).add(biases[i - 1]);
            ret.walkInOptimizedOrder(visitor);
        }

        // After all layers have been processed, ret contains the output layer.
        return ret;
    }

    /**
     * The main workhorse of the class. This method takes in training and test
     * data, along with the hyper-parameters. It trains the network then runs the
     * testing data, printing the results to the command line.
     *
     * @param trainingMatrix a RealMatrix of the training data
     * @param testMatrix a RealMatrix of the testing data
     * @param epochs the number of training/testing iterations to be run
     * @param miniBatchSize the number of training items per batch
     * @param eta the learning rate
     */
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
                int start = j * miniBatchSize;
                miniBatches[j] = trainingMatrix.getSubMatrix(
                        start,
                        start + miniBatchSize - 1,
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

    /** Runs test data through the network and identifies the number of correct
     * answers, 'correct' being that the neuron corresponding to the desired
     * result has the highest activation
     *
     * @param testData a RealMatrix of test cases
     */
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

    /**
     * Takes a batch of training examples and uses them to update the weights
     * and biases of the network.
     *
     * @param batch a RealMatrix of training examples
     * @param eta the learning rate
     */
    private void updateMiniBatch(RealMatrix batch, double eta) {
        // These are the vectors and matrices that will store the changes to the
        // weights and biases during each epoch.
        RealVector[] nabla_b = new RealVector[biases.length];
        RealVector[] delta_nabla_b = new RealVector[biases.length];
        RealMatrix[] nabla_w = new RealMatrix[weights.length];
        RealMatrix[] delta_nabla_w = new RealMatrix[weights.length];

        //Weights and biases updaters
        UpdateWeightsVisitor updateWeightsVisitor;
        UpdateBiasesVisitor updateBiasesVisitor;


        // Set the nablas.
        for(int r = 0; r < biases.length; r++){
            nabla_b[r] =
                    new ArrayRealVector(biases[r].getDimension(), 0.0);
        }
        for(int s = 0; s < weights.length; s++){
            int rows = weights[s].getRowDimension();
            int cols = weights[s].getColumnDimension();

            nabla_w[s] = new BlockRealMatrix(rows, cols);

            for(int t = 0; t < cols; t++){
                nabla_w[s].setColumnVector(t, new ArrayRealVector(rows, 0.0));
            }
        }

        //Run the backpropagation algorithm for each entry in the batch.
         for(int i = 0; i < batch.getRowDimension(); i++){
             // Set/reset the delta_nabla arrays.
             for(int r = 0; r < biases.length; r++){
                 delta_nabla_b[r] =
                         new ArrayRealVector(biases[r].getDimension(), 0.0);
             }
             for(int s = 0; s < weights.length; s++){
                 int rows = weights[s].getRowDimension();
                 int cols = weights[s].getColumnDimension();

                 delta_nabla_w[s] = new BlockRealMatrix(rows, cols);

                 for(int t = 0; t < cols; t++){
                     delta_nabla_w[s].setColumnVector(t, new ArrayRealVector(rows, 0.0));
                 }
             }

            backpropagation(delta_nabla_b, delta_nabla_w, batch.getRowVector(i));
            for(int j = 0; j < nabla_b.length; j++){
                nabla_b[j] = nabla_b[j].add(delta_nabla_b[j]);
            }
            for(int k = 0; k < nabla_w.length; k++){
                nabla_w[k] = nabla_w[k].add(delta_nabla_w[k]);
            }
        }

        //Update the weights matrices.
        for(int l = 0; l < weights.length; l++){
            updateWeightsVisitor = new UpdateWeightsVisitor(eta, batch.getRowDimension(), nabla_w[l]);
            weights[l].walkInOptimizedOrder(updateWeightsVisitor);
        }

        //Update the bias vectors.
        for(int p = 0; p < biases.length; p++){
            updateBiasesVisitor = new UpdateBiasesVisitor(eta, batch.getRowDimension(), nabla_b[p]);
            biases[p].walkInOptimizedOrder(updateBiasesVisitor);
        }
    }

    /**
     * Calculates the changes to the network's weights and biases based on a
     * single training example.
     *
     * @param delta_nabla_b an empty RealVector array for storing the changes to
     *                      the biases
     * @param delta_nabla_w an empty RealMatrix array for storing the changes to
     *                      the weights
     * @param trainingItem a RealVector storing a single training example
     */
    private void backpropagation(RealVector[] delta_nabla_b,
                                 RealMatrix[] delta_nabla_w,
                                 RealVector trainingItem) {
        // First, let's extract the label from the rest of the data and put it
        // into a vector for when we need to compute the cost derivative.
        RealVector desiredActivations = new ArrayRealVector(10, 0.0);
        desiredActivations.setEntry((int) trainingItem.getEntry(0), 1.0);

        // Stores the activations for each layer.
        RealVector[] activations = new RealVector[this.numberOfLayers];
        activations[0] = trainingItem.getSubVector(1, trainingItem.getDimension() - 1);

        // weightedInputs[i] contains the weighted inputs.
        RealVector[] weightedInputs = new RealVector[this.numberOfLayers];
        weightedInputs[0] = activations[0].copy();

        // This is just to save a bit of memory, so we don't have instantiate a
        // new Visitor every time we need one.
        SigmoidVectorVisitor sigmoidVectorVisitor = new SigmoidVectorVisitor();
        SigmoidPrimeVectorVisitor sigmoidPrimeVectorVisitor = new SigmoidPrimeVectorVisitor();

        // Temporary vectors for calculating the weightedInputs.
        RealVector z, delta;

        // Feed the training item forward through the network and store the
        // weighted inputs and activations at each layer for when we move back
        // through to calculate the error.
        for(int i = 1; i < this.numberOfLayers; i++){
            weightedInputs[i] = weights[i - 1].operate(activations[i - 1]).add(biases[i - 1]);
            activations[i] = weightedInputs[i].copy();
            activations[i].walkInOptimizedOrder(sigmoidVectorVisitor);
        }

        // Calculate the error in the final layer.
        z = weightedInputs[weightedInputs.length - 1];
        z.walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
        delta = costFunction.delta(activations[this.numberOfLayers - 1], desiredActivations, z);
        delta_nabla_b[delta_nabla_b.length - 1] = delta.copy();
        delta_nabla_w[delta_nabla_w.length - 1] =
                delta.outerProduct(activations[this.numberOfLayers - 2]);

        for(int l = delta_nabla_w.length - 2; l >= 0; l--){
            z = weightedInputs[l + 1].copy();
            z.walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
            delta = this.weights[l + 1].transpose().operate(delta).ebeMultiply(z);
            delta_nabla_b[l] = delta.copy();
            delta_nabla_w[l] = delta.outerProduct(activations[l]);
        }
    }
}
