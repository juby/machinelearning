package net.juby;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

public class SigmoidVectorVisitor implements RealVectorChangingVisitor {
    private RealVector v;

    public SigmoidVectorVisitor(RealVector v){
        this.v = v;
    }

    @Override
    public void start(int i, int i1, int i2) {
        //not used
    }

    @Override
    public double visit(int i, double v) {
        return 1.0/(1.0 + Math.exp(v));
    }

    @Override
    public double end() {
        return 0;
    }
}
