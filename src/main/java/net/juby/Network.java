package net.juby;

import org.apache.commons.math3.linear.*;

public class Network {
    private int[] layerSizes;
    private int numberOfLayers;
    //biases[i].getEntry(j) returns the bias on the (j+1)th neuron
    //  in the (i+1)th layer
    //
    //example: biases[0].getEntry(1) returns the bias on the 2nd
    //  neuron in the 1st layer
    private RealVector[] biases;
    //weights[i].getEntry(j, k) gives the weights for the connection
    //  between the (k+1)th neuron in the (i+1)th layer and the
    //  (j+1)th neuron in the (i+2)th layer.
    //
    //example: weights[1].getEntry(5, 7) returns the weight of the
    //  connection between the 8th neuron in the 2nd layer and the
    //  6th neuron in the 3rd layer
    private RealMatrix[] weights;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer
        this.layerSizes = layerSizes;
        numberOfLayers = layerSizes.length;
        biases = new RealVector[numberOfLayers];
        weights = new RealMatrix[numberOfLayers - 1];

        //initialize the vectors/matrices
        for(int i = 0; i < biases.length; i++){
            int vectorLength = layerSizes[i];
            biases[i] = new ArrayRealVector(vectorLength);
            for(int j = 0; j < vectorLength; j++){
                biases[i].setEntry(j, Math.random());
            }
        }
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
}
