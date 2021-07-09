package net.juby;

import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import static net.juby.SigmoidVectorVisitor.sigmoid;

/**
 * Applies the first derivative of the sigmoid function to a
 * {@link org.apache.commons.math3.linear.RealVector RealVector}.
 */
public class SigmoidPrimeVectorVisitor implements RealVectorChangingVisitor {
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
        // Not used
    }

    /**
     * Applies the first derivative of the sigmoid function to a single value.
     *
     * @param index the position in the RealVector this value comes from
     * @param value the input value for the sigmoid derivative
     * @return the result of the sigmoid derivative
     */
    @Override
    public double visit(int index, double value) {
        return sigmoid(value) * (1 - sigmoid(value));
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
