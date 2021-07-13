package net.juby.neuralnet.costFunctions;

import net.juby.neuralnet.visitors.SigmoidPrimeVectorVisitor;
import org.apache.commons.math3.linear.RealVector;

/**
 * Implementation of the quadratic cost function, as calculated by
 * <code>0.5*Σ||a-y||^2</code>, where <code>y</code> is the target activations
 * for the output layer and <code>a</code> is the calculated activations, for a
 * given training case.
 */
public class QuadraticCost implements CostFunction{

    @Override
    public double cost(RealVector outputActivations, RealVector desiredActivations) {
        return Math.pow(outputActivations.subtract(desiredActivations).getNorm(), 2) * 0.5;
    }

    @Override
    public RealVector delta(RealVector outputActivations, RealVector desiredActivations, RealVector finalLayerWeightedInput) {
        SigmoidPrimeVectorVisitor sigmoidPrimeVectorVisitor = new SigmoidPrimeVectorVisitor();
        RealVector temp = finalLayerWeightedInput.copy();
        temp.walkInOptimizedOrder(sigmoidPrimeVectorVisitor);

        return outputActivations.subtract(desiredActivations).ebeMultiply(temp);
    }
}
