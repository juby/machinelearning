package net.juby.exceptions;

import net.juby.neuralnet.Network;

/**
 * Exception to be thrown when the input data to {@link Network Network}
 * is not formatted correctly.
 */
public class MalformedInputDataException extends RuntimeException {

    /**
     * Creates a new exception object.
     * @param errorMessage a String of text used to give more details to the user
     */
    public MalformedInputDataException(String errorMessage){
        super(errorMessage);
    }
}
