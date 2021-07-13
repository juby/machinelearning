package net.juby.neuralnet.costFunctions;

import org.apache.commons.math3.linear.RealVector;

/**
 * Interface for abstracting out the cost function of the neural network.
 */
public interface CostFunction {
    /**
     * Calculates the cost of a training case.
     * @param outputActivations the activations in the output layer
     * @param desiredActivations the desired activations in the output layer
     * @return the cost of the given activations
     */
    double cost(RealVector outputActivations, RealVector desiredActivations);

    /**
     * Calculates the gradient of the cost function.
     * @param outputActivations the activations for a particular training case
     * @param desiredActivations the desired activations for a particular training case
     * @param finalLayerWeightedInput the weighted input for a particular training case
     * @return the gradient values for a particular training case
     */
    RealVector delta(RealVector outputActivations, RealVector desiredActivations, RealVector finalLayerWeightedInput);
}
