package br.usp.ime.bioinfogen.web.controllers;

public class MatrixFormatException
    extends
        Exception
{
    private static final long serialVersionUID = 1L;

    public MatrixFormatException(
        final String cause )
    {
        super( cause );
    }

    public MatrixFormatException(
        final Throwable ex )
    {
        super( ex );
    }
}
