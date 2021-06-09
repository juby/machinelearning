package net.juby;

import java.util.*;
import org.apache.commons.math3.linear.*;
import com.vsthost.rnd.commons.math.ext.linear.EMatrixUtils;
import net.juby.exceptions.MalformedInputDataException;
import net.juby.mnist.MnistReader;


public class Network {
    //array of the number of neurons in each layer
    private final int[] layerSizes;

    //number of layers in the network, equivalent to layerSizes.length
    private final int numberOfLayers;

    //biases[i].getEntry(j) returns the bias on the (j+1)th neuron
    //    in the (i+1)th layer.
    //
    //example: biases[0].getEntry(1) returns the bias on the 2nd
    //  neuron in the 1st layer.
    private final RealVector[] biases;

    // weights[i].getEntry(j, k) gives the weights for the connection
    //    between the (k+1)th neuron in the (i+1)th layer and the
    //    (j+1)th neuron in the (i+2)th layer.
    //
    // example: weights[1].getEntry(5, 7) returns the weight of the
    //    connection between the 8th neuron in the 2nd layer and the
    //    6th neuron in the 3rd layer.
    private final RealMatrix[] weights;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        numberOfLayers = layerSizes.length;
        biases = new RealVector[numberOfLayers];
        weights = new RealMatrix[numberOfLayers - 1];

        // Initialize the weights and biases.
        // First set all of the biases in the first layer to zero. The first
        // layer is the input layer, so no biases are needed. By setting it to
        // all zeros it makes our math a little easier later.
        for(int i = 0; i < layerSizes[0]; i++) biases[0].setEntry(i, 0.0);

        // Then create the vectors for each layer and initialize with random values.
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

    /**
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

        // Train the network using the MNIST data.
        net.stochasticGradientDescent(trainingData, trainingLabels,
                testData, testLabels, epochs, miniBatchSize, eta);
    }

    private RealVector feedForward(RealVector input){
        // The first layer is the input layer.
        RealVector ret = input.copy();

        // For each layer, calculate a' = σ(wa+b).
        // [The operate() method multiplies the matrix by a given vector.]
        for(int i = 0; i < numberOfLayers; i++){
            weights[i].operate(ret).add(biases[i])
                    .walkInOptimizedOrder(new SigmoidVectorVisitor());
        }

        // After all layers have been processed, ret contains the output layer.
        return ret;
    }

    private void stochasticGradientDescent(List<int[][]> trainingData,
                                          int[] trainingLabels,
                                          List<int[][]> testData,
                                          int[] testLabels,
                                          int epochs,
                                          int miniBatchSize,
                                          double eta){

        // Local variable setup.
        int nTest = testLabels.length;
        int miniBatchCount = trainingLabels.length/miniBatchSize;
        RealMatrix[] miniBatches = new RealMatrix[miniBatchCount];

        // Convert the training and test data from a List of 2D matrices into a
        // RealMatrix, where the first column is the label and the remaining columns
        // are the image data laid out such that each row begins in the position
        // following the last entry of the previous row. We do this so we can
        // easily shuffle the rows; later we can use utility methods to extract
        // submatricies in order to do the necessary linear algebra.
        RealMatrix trainingMatrix =
                new Array2DRowRealMatrix(trainingData.size(), this.layerSizes[0] + 1);
        for(int i = 0; i < trainingData.size(); i++){
            int[][] tempAry = trainingData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols; k++){
                    double entryValue;

                    if(k == 0) entryValue = trainingLabels[k];
                    else entryValue = tempAry[j][k];

                    trainingMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }

        RealMatrix testMatrix =
                new Array2DRowRealMatrix(testData.size(), this.layerSizes[0] + 1);
        for(int i = 0; i < testData.size(); i++){
            int[][] tempAry = testData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;
            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols; k++){
                    double entryValue;

                    if(k == 0) entryValue = testLabels[k];
                    else entryValue = tempAry[j][k];

                    testMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }

        // Run this loop for each epoch.
        for(int i = 0; i < epochs; i++) {
            //Randomize the training data.
            trainingMatrix = EMatrixUtils.shuffleRows(trainingMatrix);

            //Generate the mini batches.
            for (int j = 0; j < miniBatchCount; j++) {
                miniBatches[j] = trainingMatrix.getSubMatrix(
                        j * miniBatchCount,
                        j * (miniBatchCount + 1) - 1,
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
        int total = 0;

        // Loop through all of the test data
        for(int i = 0; i < testData.getRowDimension(); i++){
            int targetValue, resultValue;
            targetValue = (int) testData.getEntry(i, 0);

            // Maybe there's a more elegant way to do this, but for now we need
            // to   1)pull everything but the first column into a matrix
            //      2)turn that matrix into a vector
            //      3)feed that vector through the net
            //      4)then pull the maximum value
            resultValue = (int) feedForward(
                    testData.getSubMatrix(
                            i,
                            i,
                            1,
                            testData.getColumnDimension()
                    ).getRowVector(0)
            ).getMaxValue();

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
        // I'm not a fan of these variable names, but at this point in the book
        // the backpropagation algorithm hasn't really been explained. Once I
        // have a better understanding I'll likely rename these variables to
        // something a bit more intuitive.
        RealVector[] nabla_b = new RealVector[biases.length];
        RealVector[] delta_nabla_b = new RealVector[biases.length];
        RealMatrix[] nabla_w = new RealMatrix[weights.length];
        RealMatrix[] delta_nabla_w = new RealMatrix[weights.length];

        //todo refactor this with new data structures in mind

        // Set up the nablas
        for(int i = 0; i < biases.length; i++){
            nabla_b[i] =
                    new ArrayRealVector(biases[i].getDimension(), 0.0);
            delta_nabla_b[i] =
                    new ArrayRealVector(biases[i].getDimension(), 0.0);
        }
        for(int i = 0; i < weights.length; i++){
            int rows = weights[i].getRowDimension();
            int cols = weights[i].getColumnDimension();
            RealVector temp = new ArrayRealVector(rows, 0.0);

            nabla_w[i] = new Array2DRowRealMatrix(rows, cols);
            delta_nabla_w[i] = new Array2DRowRealMatrix(rows, cols);

            for(int j = 0; j < cols; j++){
                nabla_w[i].setColumnVector(j, temp);
                delta_nabla_w[i].setColumnVector(j, temp);
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

    private void backpropagation(RealVector[] delta_nabla_b, RealMatrix[] delta_nabla_w, RealVector rowVector) {
        //todo: backpropagation
    }

    //todo costderivative
    //todo sigmoidprime - may just make this another Visitor
}
