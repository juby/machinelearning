package net.juby;

import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.linear.FieldVectorChangingVisitor;
import org.apache.commons.math3.util.BigReal;

public class SigmoidVectorVisitor<T extends FieldElement<BigReal>> implements FieldVectorChangingVisitor<T>{
    @Override
    public void start(int dimension, int start, int end) {
        // Not used.
    }

    @Override
    public T end() {
        return null;
    }

    @Override
    public T visit(int index, T value) {
        return (T) sigmoid((BigReal) value);
    }

    protected static BigReal sigmoid(BigReal value){
        return new BigReal(1.0/(1.0 + Math.exp(-value.doubleValue())));
    }
}
