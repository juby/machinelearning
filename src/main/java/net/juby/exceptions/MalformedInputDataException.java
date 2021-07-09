package net.juby.exceptions;

/**
 * Exception to be thrown when the input data to {@link net.juby.Network Network}
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
