package net.juby.neuralnet.visitors;

import org.apache.commons.math3.linear.RealVectorChangingVisitor;

/**
 * Applies the sigmoid function to a
 * {@link org.apache.commons.math3.linear.RealVector RealVector}.
 */
public class SigmoidVectorVisitor implements RealVectorChangingVisitor {

    /**
     * Runs when the visitor is first invoked. Does nothing, only included
     * as part of the interface implementation.
     *
     * @param dimension the number of items in the RealVector
     * @param start the index to start traversing from
     * @param end the index to stop traversing at
     */
    @Override
    public void start(int dimension, int start, int end) {
        //not used
    }

    /**
     * Applies the sigmoid function to a single input.
     *
     * @param value the input for the sigmoid function
     * @return the result of the sigmoid function
     */
    protected static double sigmoid(double value) {
        return 1.0/(1.0 + Math.exp(-value));
    }

    /**
     * Applies the sigmoid function to a single value from a RealVector.
     *
     * @param index the position in the RealVector this value comes from
     * @param value the input for the sigmoid function
     * @return the result of the sigmoid function
     */
    @Override
    public double visit(int index, double value) {
        return sigmoid(value);
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
