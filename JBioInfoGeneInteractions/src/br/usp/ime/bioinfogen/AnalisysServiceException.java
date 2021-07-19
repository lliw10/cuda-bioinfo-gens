package br.usp.ime.bioinfogen;

public class AnalisysServiceException
    extends
        Exception
{
    private static final long serialVersionUID = 1L;

    public AnalisysServiceException(
        final Throwable ex )
    {
        super( ex );
    }
}
