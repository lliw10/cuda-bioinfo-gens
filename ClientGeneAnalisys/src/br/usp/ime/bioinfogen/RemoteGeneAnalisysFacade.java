package br.usp.ime.bioinfogen;

import java.io.Serializable;
import java.util.List;

import org.apache.log4j.Logger;

import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

public class RemoteGeneAnalisysFacade
    implements
        Serializable
{
    private final Logger logger = Logger.getLogger( getClass() );
    private static final long serialVersionUID = 7430512039736945837L;

    private final GeneAnalisysExecutorFacade executor;

    public RemoteGeneAnalisysFacade(
        final GeneAnalisysExecutorFacade executor )
    {
        this.executor = executor;
        logger.info( "RemoteGeneAnalisysFacade started" );
    }

    public void execute(
        final ExecutionConfiguration configuration )
        throws AnalisysServiceException
    {
        logger.info( "Requesting execution of task:" + configuration.getTaskId() );
        executor.execute( configuration );
    }

    public byte[] getInputFile(
        final String taskId )
        throws AnalisysServiceException
    {
        return executor.getInputFile( taskId );
    }

    public List<JBioinfoTask> listAllTasks()
    {
        return executor.getTasks();
    }

    public JBioinfoTask getTask(
        final String taskId )
    {
        return executor.getTask( taskId );
    }

}
