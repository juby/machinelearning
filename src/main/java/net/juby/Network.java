package net.juby;

import org.apache.commons.math3.linear.*;
import com.vsthost.rnd.commons.math.ext.linear.EMatrixUtils;

public class Network {
    private int[] layerSizes; //array of the number of neurons in each layer
    private int numberOfLayers; //number of layers in the network, equivalent to layerSizes.length
    //biases[i].getEntry(j) returns the bias on the (j+1)th neuron
    //  in the (i+1)th layer.
    //
    //example: biases[0].getEntry(1) returns the bias on the 2nd
    //  neuron in the 1st layer.
    private RealVector[] biases;
    //weights[i].getEntry(j, k) gives the weights for the connection
    //  between the (k+1)th neuron in the (i+1)th layer and the
    //  (j+1)th neuron in the (i+2)th layer.
    //
    //example: weights[1].getEntry(5, 7) returns the weight of the
    //  connection between the 8th neuron in the 2nd layer and the
    //  6th neuron in the 3rd layer.
    private RealMatrix[] weights;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer.
        this.layerSizes = layerSizes;
        numberOfLayers = layerSizes.length;
        biases = new RealVector[numberOfLayers];
        weights = new RealMatrix[numberOfLayers - 1];

        //Initialize the vectors/matrices
        //First set all of the biases in the first layer to zero. The first layer is the input layer, so no biases are
        //  needed. By setting it to all zeros it makes our math a little easier later.
        for(int i = 0; i < layerSizes[0]; i++) biases[0].setEntry(i, 0.0);
        //Then create the vectors for each layer and initialize with random values.
        for(int i = 1; i < biases.length; i++){
            int vectorLength = layerSizes[i];
            biases[i] = new ArrayRealVector(vectorLength);
            for(int j = 0; j < vectorLength; j++){
                biases[i].setEntry(j, Math.random());
            }
        }
        //Finally create the weights matrices and initialize with random values.
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

    public RealVector feedForward(RealVector input){
        //The first layer is the input layer.
        RealVector ret = input.copy();

        //For each layer, calculate a' = σ(wa+b).
        for(int i = 0; i < numberOfLayers; i++){
            weights[i].operate(ret).add(biases[i]).walkInOptimizedOrder(new SigmoidVectorVisitor());
        }

        //After all layers have been processed, ret contains the output layer.
        return ret;
    }

    public void stochasticGradientDescent(RealMatrix trainingData,
                                          int epochs,
                                          int miniBatchSize,
                                          double eta){
        stochasticGradientDescent(trainingData, epochs, miniBatchSize, eta, null);
    }

    public void stochasticGradientDescent(RealMatrix trainingData,
                                          int epochs,
                                          int miniBatchSize,
                                          double eta,
                                          RealMatrix testData){
        //Local variable setup.
        int nTest = -1;
        int miniBatchCount = trainingData.getRowDimension()/miniBatchSize;
        RealMatrix[] miniBatches = new RealMatrix[miniBatchCount];
        if(testData != null) nTest = testData.getRowDimension();
        int n = trainingData.getRowDimension();
        double[][] temp = new double[miniBatchSize][trainingData.getColumnDimension()];

        //Run this loop for each epoch.
        for(int i = 0; i < epochs; i++) {
            //Randomize the training data.
            trainingData = EMatrixUtils.shuffleRows(trainingData);

            //Generate the mini batches.
            for (int j = 0; j < miniBatchCount; j++) {
                trainingData.copySubMatrix(j * miniBatchCount,
                        j * (miniBatchCount + 1) - 1,
                        0,
                        trainingData.getColumnDimension() - 1,
                        temp);
                miniBatches[j] = MatrixUtils.createRealMatrix(temp);
            }

            //Run the mini batches.
            for (RealMatrix batch : miniBatches) {
                updateMiniBatch(batch, eta);
            }

            //Output progress to command line.
            if(testData != null){
                System.out.println("Epoch " + i + ": " + evaluate(testData) + "/" + nTest);
            } else {
                System.out.println("Epoch " + i + " complete.");
            }
        }
    }

    private double evaluate(RealMatrix testData) {
        //todo evaluate
        return 0.0;
    }

    private void updateMiniBatch(RealMatrix batch, double eta) {
        //I'm not a fan of these variable names, but at this point in the book the backpropagation algorithm hasn't
        //really been explained. Once I have a better understanding I'll likely rename these variables to something
        //a bit more intuitive.
        RealVector[] nabla_b = new RealVector[biases.length];
        RealMatrix[] nabla_w = new RealMatrix[weights.length];

        //Set up the nablas
        for(int i = 0; i < biases.length; i++){
            nabla_b[i] = new ArrayRealVector(biases[i].getDimension(), 0.0);
        }
        for(int i = 0; i < weights.length; i++){
            int rows = weights[i].getRowDimension();
            int cols = weights[i].getColumnDimension();
            RealVector temp = new ArrayRealVector(rows, 0.0);

            nabla_w[i] = new Array2DRowRealMatrix(rows, cols);

            for(int j = 0; j < cols; j++){
                nabla_w[i].setColumnVector(j, temp);
            }
        }

        //todo updateMiniBatch
    }

    //todo backpropagation
    //todo costderivative
    //todo sigmoidprime - may just make this another Visitor
}
