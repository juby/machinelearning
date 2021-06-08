package net.juby.exceptions;

public class MalformedTestDataException extends RuntimeException {
    public MalformedTestDataException(String errorMessage){
        super(errorMessage);
    }
}
