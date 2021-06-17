package net.juby;

import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import static net.juby.SigmoidVectorVisitor.sigmoid;

public class SigmoidPrimeVectorVisitor implements RealVectorChangingVisitor {
    @Override
    public void start(int dimension, int start, int end) {
        // Not used
    }

    @Override
    public double visit(int index, double value) {
        return sigmoid(value) * (1 - sigmoid(value));
    }

    @Override
    public double end() {
        return 0;
    }
}
