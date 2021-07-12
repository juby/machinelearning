package net.juby.costFunctions;

import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.linear.*;
import static com.vsthost.rnd.commons.math.ext.linear.DMatrixUtils.sum;

/**
 * Implementation of the cross-entropy cost function, as calculated by
 * <code>-y*(log a)-(1-y)*log(1-a)</code>, where <code>y</code> is the target output for a
 * test case, <code>a</code> is the calculated output for that test case, and
 * <code>*</code> represents the
 * <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)">Hadamard product</a>.
 */
public class CrossEntropyCost implements CostFunction{
    @Override
    public double cost(RealVector outputActivations, RealVector desiredActivations) {
        return sum(
                (
                    (desiredActivations.mapMultiply(-1).ebeMultiply(outputActivations.map(new Log())))
                    .ebeMultiply(
                            (new ArrayRealVector(outputActivations.getDimension(), 1.0))
                                .subtract(outputActivations))
                ).toArray()
        );
    }

    /**
     * Calculates the gradient of the cost functions.
     * @param outputActivations the activations for a particular training case
     * @param desiredActivations the desired activations for a particular training case
     * @param finalLayerWeightedInput not used by this implementation, can be
     *                                simply set to <code>null</code>.
     * @return the gradient values for a particular training case
     */
    @Override
    public RealVector delta(RealVector outputActivations, RealVector desiredActivations, RealVector finalLayerWeightedInput) {
        return outputActivations.subtract(desiredActivations);
    }
}
