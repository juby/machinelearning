package net.juby;

import org.apache.commons.math3.linear.*;

public class Network {
    private final int[] layerSizes;
    private final int numberOfLayers;

    //biases[i].getEntry(j) returns the bias on the (j+1)th neuron
    //  in the (i+1)th layer
    //
    //example: biases[0].getEntry(1) returns the bias on the 2nd
    //  neuron in the 1st layer
    private RealVector[] biases;

    //weights[i].getEntry(j, k) gives the weights for the connection
    //  between the (j+1)th neuron in the (i+1)th layer and the
    //  (k+1)th neuron in the (i+2)th layer.
    //
    //example: weights[1].getEntry(5, 7) returns the weight of the
    //  connection between the 6th neuron in the 2nd layer and the
    //  8th neuron in the 3rd layer
    private RealMatrix[] weights;

    public Network(int[] layerSizes){
        //Set the number of layers and the size of each layer
        this.layerSizes = layerSizes;
        numberOfLayers = layerSizes.length;

        biases = new RealVector[numberOfLayers];
        weights = new RealMatrix[numberOfLayers - 1];

        //TODO: initialize matricies and arrays and fill them with random values
    }
}
