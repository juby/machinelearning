package net.juby.visitors;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;

public class UpdateWeightsVisitor implements RealMatrixChangingVisitor {
    private final double eta;
    private final int batchSize;
    private final RealMatrix nabla_w;

    public UpdateWeightsVisitor(double eta, int batchSize, RealMatrix nabla_w){
        this.eta = eta;
        this.batchSize = batchSize;
        this.nabla_w = nabla_w;
    }

    @Override
    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
        //not used
    }

    @Override
    public double visit(int row, int column, double value) {
        return value - ((eta/batchSize) * nabla_w.getEntry(row, column));
    }

    @Override
    public double end() {
        return 0;
    }
}
