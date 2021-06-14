package net.juby;

import java.util.*;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.linear.*;
import com.vsthost.rnd.commons.math.ext.linear.EMatrixUtils;
import net.juby.exceptions.MalformedInputDataException;
import net.juby.mnist.MnistReader;


public class Network {
    private int[] layerSizes;
    private int numberOfLayers;
    private RealVector[] biases;
    private RealMatrix[] weights;

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
        RealMatrix trainingMatrix = new Array2DRowRealMatrix(trainingData.size(),
                        this.layerSizes[0] + 1);
        for(int i = 0; i < trainingData.size(); i++){
            int[][] tempAry = trainingData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    double entryValue;

                    if(k == 0) entryValue = trainingLabels[k];
                    else entryValue = tempAry[j][k - 1];

                    trainingMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }

        // Generate a RealMatrix of the test data.
        RealMatrix testMatrix =
                new Array2DRowRealMatrix(testData.size(), this.layerSizes[0] + 1);
        for(int i = 0; i < testData.size(); i++){
            int[][] tempAry = testData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;
            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    double entryValue;

                    if(k == 0) entryValue = testLabels[k];
                    else entryValue = tempAry[j][k - 1];

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
        int total, targetValue, resultValue;
        total = 0;

        // Loop through all of the test data
        for(int i = 0; i < testData.getRowDimension(); i++){
            targetValue = (int) testData.getEntry(i, 0);

            // Extract the test case from the matrix, feed it through, and get
            // the max index (as neuron 0 represents the neuron with the net's
            // 'guess' at how likely it's a 0, and so on).
            resultValue = feedForward(
                    new ArrayRealVector(testData.getRow(i), 1, testData.getColumnDimension())
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
        // I'm not a fan of these variable names, but at this point in the book
        // the backpropagation algorithm hasn't really been explained. Once I
        // have a better understanding I'll likely rename these variables to
        // something a bit more intuitive.
        RealVector[] nabla_b = new RealVector[biases.length];
        RealVector[] delta_nabla_b = new RealVector[biases.length];
        RealMatrix[] nabla_w = new RealMatrix[weights.length];
        RealMatrix[] delta_nabla_w = new RealMatrix[weights.length];

        // Set up the nablas
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
                                 RealVector rowVector) {
        //todo: backpropagation
    }

    //todo costderivative
    //todo sigmoidprime - may just make this another Visitor
}
