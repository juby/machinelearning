package net.juby.exceptions;

public class MalformedInputDataException extends RuntimeException {
    public MalformedInputDataException(String errorMessage){
        super(errorMessage);
    }
}
