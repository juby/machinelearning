package net.juby.visitors;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

public class UpdateBiasesVisitor implements RealVectorChangingVisitor {
    private double eta;
    private int batchSize;
    private RealVector nabla_b;

    public UpdateBiasesVisitor(double eta, int batchSize, RealVector nabla_b){
        this.eta = eta;
        this.batchSize = batchSize;
        this.nabla_b = nabla_b;
    }

    @Override
    public void start(int dimension, int start, int end) {
        //not used
    }

    @Override
    public double visit(int index, double value) {
        return value - ((eta/batchSize) * nabla_b.getEntry(index));
    }

    @Override
    public double end() {
        return 0;
    }
}
