package net.juby;

import net.juby.exceptions.MalformedInputDataException;
import net.juby.mnist.MnistReader;
import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.BigReal;
import org.apache.commons.math3.util.MathArrays;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Network {
    private final int numberOfLayers;
    private final List<FieldVector<BigReal>> biases;
    private final List<FieldMatrix<BigReal>> weights;
    private final int[] layerSizes;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes.clone();
        this.numberOfLayers = layerSizes.length;
        this.biases = new ArrayList<>(numberOfLayers);
        this.weights = new ArrayList<>(numberOfLayers - 1);

        // Initialize the weights and biases.

        // Create the vectors for each layer and initialize with random values.
        // biases[i] contains the biases for the (i+1)th layer.
        // We create a vector of biases for the 1st layer (biases[0]) but we
        // won't actually be using it. It's just there to make things a little
        // cleaner.
        for(int i = 0; i < biases.size(); i++){
            int vectorLength = layerSizes[i];
            biases.set(i, new ArrayFieldVector<>(vectorLength, new BigReal(Math.random())));
        }
        // Finally create the weights matrices and initialize with random values.
        // weights[i] contains the weights connecting the (i+1)th layer to the
        // (i+2)th layer.
        for(int i = 0; i < weights.size(); i++){
            int cols = layerSizes[i];
            int rows = layerSizes[i + 1];
            weights.set(i, new BlockFieldMatrix<>(new BigReal[rows][cols]));
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    weights.get(i).setEntry(j, k, new BigReal(Math.random()));
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
        FieldMatrix<BigReal> trainingMatrix, testMatrix;
        trainingMatrix = new BlockFieldMatrix<>(BigReal.ZERO.getField(),
                trainingData.size(), net.layerSizes[0] + 1);
        testMatrix = new BlockFieldMatrix<>(BigReal.ZERO.getField(),
                testData.size(), net.layerSizes[0] + 1);

        convertData(trainingData, trainingLabels, testData, testLabels, trainingMatrix, testMatrix);

        // Train the network using the MNIST data.
        net.stochasticGradientDescent(trainingMatrix, testMatrix,
                epochs, miniBatchSize, new BigReal(eta));
    }

    private static void convertData(List<int[][]> trainingData,
                             int[] trainingLabels,
                             List<int[][]> testData,
                             int[] testLabels,
                             FieldMatrix<BigReal> trainingMatrix,
                             FieldMatrix<BigReal> testMatrix){
        // Convert the training and test data from a List of 2D matrices into a
        // FieldMatrix, where the first column is the label and the remaining columns
        // are the image data laid out such that each row begins in the position
        // following the last entry of the previous row. We do this so we can
        // easily shuffle the rows; later we can use utility methods to extract
        // submatrices in order to do the necessary linear algebra.
        BigReal entryValue;

        for(int i = 0; i < trainingData.size(); i++){
            int[][] tempAry = trainingData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;

            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    // I don't think that normalizing the entries to be between
                    // 0 and 1 (inclusive) is strictly necessary, but it doesn't
                    // hurt.
                    if(k == 0) entryValue = new BigReal(trainingLabels[k]);
                    else entryValue = new BigReal(tempAry[j][k - 1]/255.0);

                    trainingMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }

        // Generate a FieldMatrix of the test data.
        for(int i = 0; i < testData.size(); i++){
            int[][] tempAry = testData.get(i);
            int tempAryRows = tempAry.length;
            int tempAryCols = tempAry[0].length;
            for(int j = 0; j < tempAryRows; j++){
                for(int k = 0; k < tempAryCols + 1; k++){
                    if(k == 0) entryValue = new BigReal(testLabels[k]);
                    else entryValue = new BigReal(tempAry[j][k - 1]/255.0);

                    testMatrix.setEntry(i, j*tempAryCols + k, entryValue);
                }
            }
        }
    }

    private FieldVector<BigReal> feedForward(FieldVector<BigReal> input){
        // The first layer is the input layer.
        FieldVector<BigReal> ret = new ArrayFieldVector<>(input);
        SigmoidVectorVisitor<BigReal> visitor = new SigmoidVectorVisitor<>();

        // For each layer, calculate a' = σ(wa+b).
        // [The operate() method multiplies the matrix by a given vector.]
        for(int i = 1; i < numberOfLayers; i++){
            ret = weights.get(i - 1).operate(ret).add(biases.get(i));
            ((ArrayFieldVector<BigReal>) ret).walkInOptimizedOrder(visitor);
        }

        // After all layers have been processed, ret contains the output layer.
        return ret;
    }

    private void stochasticGradientDescent(FieldMatrix<BigReal> trainingMatrix,
                                          FieldMatrix<BigReal> testMatrix,
                                          int epochs,
                                          int miniBatchSize,
                                          BigReal eta){
        // Local variable setup.
        int nTest = testMatrix.getRowDimension();
        int miniBatchCount = trainingMatrix.getRowDimension()/miniBatchSize;
        List<FieldMatrix<BigReal>> miniBatches = new ArrayList<>(miniBatchCount);

        // Run this loop for each epoch.
        for(int i = 0; i < epochs; i++) {
            //Randomize the training data.
            trainingMatrix = shuffleRows(trainingMatrix);

            //Generate the mini batches.
            for (int j = 0; j < miniBatchCount; j++) {
                int start = j * miniBatchSize;
                miniBatches.set(j, trainingMatrix.getSubMatrix(
                        start,
                        start + miniBatchSize - 1,
                        0,
                        trainingMatrix.getColumnDimension() - 1
                ));
            }

            // Run the mini batches.
            for (FieldMatrix<BigReal> batch : miniBatches) {
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
    private int evaluate(FieldMatrix<BigReal> testData) {
        int total, targetValue, resultValue;
        total = 0;

        // Loop through all of the test data
        for(int i = 0; i < testData.getRowDimension(); i++){
            targetValue = (int) testData.getEntry(i, 0).doubleValue();

            // Extract the test case from the matrix, feed it through, and get
            // the max index (as neuron 0 represents the neuron with the net's
            // 'guess' at how likely it's a 0, and so on).
            resultValue = getResult(
                    feedForward(
                        new ArrayFieldVector<>(testData.getRow(i), 1,
                            testData.getColumnDimension() - 1)
                    )
            );

            // Add to the running tally if it's a correct answer.
            if(targetValue == resultValue) total += 1;
        }
        return total;
    }

    // From the sample code in the textbook:
    // Update the network's weights and biases by applying gradient descent
    // using backpropagation to a single mini batch. The "batch" is a list of
    // tuples "(x, y)", and "eta" is the learning rate.
    private void updateMiniBatch(FieldMatrix<BigReal> batch, BigReal eta) {
        // These are the vectors and matrices that will store the changes to the
        // weights and biases during each epoch.
        List<FieldVector<BigReal>> nabla_b = new ArrayList<>(biases.size());
        List<FieldVector<BigReal>> delta_nabla_b = new ArrayList<>(biases.size());
        List<FieldMatrix<BigReal>> nabla_w = new ArrayList<>(weights.size());
        List<FieldMatrix<BigReal>> delta_nabla_w = new ArrayList<>(weights.size());

        // Set the nablas.
        for(int r = 0; r < biases.size(); r++){
            nabla_b.set(r, new ArrayFieldVector<>(biases.get(r).getDimension(), BigReal.ZERO));
        }
        for(int s = 0; s < weights.size(); s++){
            int rows = weights.get(s).getRowDimension();
            int cols = weights.get(s).getColumnDimension();

            nabla_w.set(s, new BlockFieldMatrix<>(BigReal.ZERO.getField(), rows, cols));

            for(int t = 0; t < cols; t++){
                nabla_w.get(s).setColumnVector(t,
                        new ArrayFieldVector<>(rows, BigReal.ZERO));
            }
        }

        //Run the backpropagation algorithm for each entry in the batch.
        // Set/reset the delta_nabla arrays.
        for(int r = 0; r < biases.size(); r++){
            delta_nabla_b.set(r, new ArrayFieldVector<>(biases.get(r).getDimension(), BigReal.ZERO));
        }
        for(int s = 0; s < weights.size(); s++){
            int rows = weights.get(s).getRowDimension();
            int cols = weights.get(s).getColumnDimension();

            delta_nabla_w.set(s, new BlockFieldMatrix<BigReal>(BigReal.ZERO.getField(), rows, cols));

            for(int t = 0; t < cols; t++){
                delta_nabla_w.get(s).setColumnVector(t, new ArrayFieldVector<>(rows, BigReal.ZERO));
            }
        }

        for(int i = 0; i < batch.getRowDimension(); i++){
            backpropagation(delta_nabla_b, delta_nabla_w, batch.getRowVector(i));
            for(int j = 0; j < nabla_b.size(); j++){
                nabla_b.get(j).add(delta_nabla_b.get(j));
            }
            for(int k = 0; k < nabla_w.size(); k++){
                nabla_w.get(k).add(delta_nabla_w.get(k));
            }
        }

        //Update the weights matrices.
        for(int l = 0; l < weights.size(); l++){
            for(int m = 0; m < weights.get(l).getRowDimension(); m++){
                for(int n = 0; n < weights.get(l).getColumnDimension(); n++){
                    BigReal current = (BigReal) weights.get(l).getEntry(m, n);
                    BigReal delta = eta
                                    .divide(new BigReal(batch.getRowDimension()))
                                    .multiply(nabla_w.get(l).getEntry(m, n));
                    weights.get(l).setEntry(m, n, current.subtract(delta));
                }
            }
        }

        //Update the bias vectors.
        for(int p = 0; p < biases.size(); p++){
            for(int q = 0; q < biases.get(p).getDimension(); q++){
                BigReal current = biases.get(p).getEntry(q);
                //double delta = (eta/batch.getRowDimension()) * nabla_b[p].getEntry(q);
                BigReal delta = eta
                                .divide(new BigReal(batch.getRowDimension()))
                                .multiply(nabla_b.get(p).getEntry(q));
                biases.get(p).setEntry(q, current.subtract(delta));
            }
        }
    }

    private void backpropagation(List<FieldVector<BigReal>> delta_nabla_b,
                                 List<FieldMatrix<BigReal>> delta_nabla_w,
                                 FieldVector<BigReal> trainingItem) {
        // First, let's extract the label from the rest of the data and put it
        // into a vector for when we need to compute the cost derivative.
        ArrayFieldVector<BigReal> desiredActivations = new ArrayFieldVector<BigReal>(10, BigReal.ZERO);
        desiredActivations.setEntry((int) trainingItem.getEntry(0).doubleValue(), BigReal.ONE);

        // Stores the activations for each layer.
        List<FieldVector<BigReal>> activations = new ArrayList<>(this.numberOfLayers);
        activations.set(0, trainingItem.getSubVector(1, trainingItem.getDimension() - 1));

        // weightedInputs[i] contains the weighted inputs.
        List<FieldVector<BigReal>> weightedInputs = new ArrayList<>(this.numberOfLayers);
        weightedInputs.set(0, activations.get(0).copy());

        // This is just to save a bit of memory, so we don't have instantiate a
        // new Visitor every time we need one.
        SigmoidVectorVisitor<BigReal> sigmoidVectorVisitor = new SigmoidVectorVisitor<>();
        SigmoidPrimeVectorVisitor<BigReal> sigmoidPrimeVectorVisitor = new SigmoidPrimeVectorVisitor<>();

        // Temporary vectors for calculating the weightedInputs.
        FieldVector<BigReal> z, delta;

        // Feed the training item forward through the network and store the
        // weighted inputs and activations at each layer for when we move back
        // through to calculate the error.
        for(int i = 1; i < this.numberOfLayers; i++){
            weightedInputs.set(i, weights.get(i - 1)
                    .operate(activations.get(i - 1)).add(biases.get(i)));
            activations.set(i, weightedInputs.get(i).copy());
            ((ArrayFieldVector<BigReal>) activations.get(i))
                    .walkInOptimizedOrder(sigmoidVectorVisitor);
        }

        // Calculate the error in the final layer.
        z = weightedInputs.get(weightedInputs.size() - 1);
        ((ArrayFieldVector<BigReal>) z).walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
        delta = costDerivative(activations
                        .get(this.numberOfLayers - 1), desiredActivations)
                .ebeMultiply(z);
        delta_nabla_b.set(delta_nabla_b.size() - 1, delta.copy());
        delta_nabla_w.set(delta_nabla_w.size() - 1,
                delta.outerProduct(activations.get(this.numberOfLayers - 2)));

        for(int l = delta_nabla_w.size() - 2; l >= 0; l--){
            z = weightedInputs.get(l + 1).copy();
            ((ArrayFieldVector<BigReal>) z).walkInOptimizedOrder(sigmoidPrimeVectorVisitor);
            delta = this.weights.get(l + 1).transpose().operate(delta).ebeMultiply(z);
            delta_nabla_b.set(l, delta.copy());
            delta_nabla_w.set(l, delta.outerProduct(activations.get(l)));
        }
    }

    private static FieldVector<BigReal> costDerivative(FieldVector<BigReal> outputActivations, FieldVector<BigReal> desiredActivations){
        return outputActivations.subtract(desiredActivations);
    }

    private static int getResult(FieldVector<BigReal> output){
        int ret = 0;

        for (int i = 1; i < output.getDimension(); i++){
            BigReal t1, t2;
            t1 = output.getEntry(ret);
            t2 = output.getEntry(i);
            if(t2.compareTo(t1) > 0) ret = i;
        }

        return ret;
    }

    // The below code is adapted from the Apache Commons Math Extensions.
    // Available at https://github.com/vst/commons-math-extensions.

    /*
     * Copyright (c) 2015 Vehbi Sinan Tunalioglu <vst@vsthost.com>, Tolga Sezer <tolgasbox@gmail.com>.
     *
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     *     http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */


    /**
     * Shuffles rows of a matrix.
     *
     * @param matrix The matrix of which the rows will be shuffled.
     * @return The new shuffled matrix.
     */
    private static <T extends FieldElement<T>> FieldMatrix<T> shuffleRows (FieldMatrix<T> matrix) {
        return shuffleRows(matrix, new MersenneTwister());
    }

    /**
     * Shuffles rows of a matrix using the provided random number generator.
     *
     * @param matrix The matrix of which the rows will be shuffled.
     * @param randomGenerator The random number generator to be used.
     * @return The new shuffled matrix.
     */
    private static <T extends FieldElement<T>> FieldMatrix<T> shuffleRows  (FieldMatrix<T> matrix, RandomGenerator randomGenerator) {
        // Create an index vector to be shuffled:
        int[] index = MathArrays.sequence(matrix.getRowDimension(), 0, 1);
        MathArrays.shuffle(index, randomGenerator);

        // Create a new matrix:
        FieldMatrix<T> retval = MatrixUtils.createFieldMatrix(matrix.getField(), matrix.getRowDimension(), matrix.getColumnDimension());

        // Populate:
        for (int row = 0; row < index.length; row++) {
            retval.setRowVector(row, matrix.getRowVector(index[row]));
        }

        // Done, return:
        return retval;
    }
}
