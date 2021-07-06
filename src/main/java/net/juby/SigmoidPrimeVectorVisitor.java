package net.juby;

import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.linear.FieldVectorChangingVisitor;
import org.apache.commons.math3.util.BigReal;
import static net.juby.SigmoidVectorVisitor.sigmoid;

public class SigmoidPrimeVectorVisitor<T extends FieldElement<BigReal>> implements FieldVectorChangingVisitor<T> {

    @Override
    public void start(int dimension, int start, int end) {
        // Not used.
    }

    @Override
    public T visit(int index, T value) {
        return (T) (sigmoid((BigReal) value)).multiply((sigmoid((BigReal) value)).negate().add(BigReal.ONE));
    }

    @Override
    public T end() {
        return null;
    }
}
