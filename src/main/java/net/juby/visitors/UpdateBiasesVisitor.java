package net.juby.visitors;

import net.juby.neuralnet.Network;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

/**
 * Updates the {@link RealVector} of weights in a {@link Network}.
 */
public class UpdateBiasesVisitor implements RealVectorChangingVisitor {
    private final double eta;
    private final int batchSize;
    private final RealVector nabla_b;

    /**
     * Initializes the updater.
     * @param eta learning rate for the network
     * @param batchSize number of training cases per batch
     * @param nabla_b the gradient values for a batch of training cases
     */
    public UpdateBiasesVisitor(double eta, int batchSize, RealVector nabla_b){
        this.eta = eta;
        this.batchSize = batchSize;
        this.nabla_b = nabla_b;
    }

    /**
     * Runs when the visitor is first invoked. Does nothing, only included
     * as part of the interface implementation.
     * @param dimension number of rows in the vector
     * @param start index to start at
     * @param end index to end at
     */
    @Override
    public void start(int dimension, int start, int end) {
        //not used
    }

    /**
     * Updates a single value in the biases vector.
     * @param index the position of the value being updated
     * @param value the original value from the vector
     * @return the updated value
     */
    @Override
    public double visit(int index, double value) {
        return value - ((eta/batchSize) * nabla_b.getEntry(index));
    }

    /**
     * Runs when the visitor is finished traversing the RealVector. Does nothing,
     * only included as part of the interface implementation.
     * @return the floating point value 0
     */
    @Override
    public double end() {
        return 0;
    }
}
