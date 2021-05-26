package net.juby;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;

public class SigmoidMatrixVisitor implements RealMatrixChangingVisitor {
    @Override
    public void start(int i, int i1, int i2, int i3, int i4, int i5) {
        //not used
    }

    @Override
    public double visit(int i, int i1, double v) {
        return 1.0/(1.0 + Math.exp(v));
    }

    @Override
    public double end() {
        return 0;
    }
}
