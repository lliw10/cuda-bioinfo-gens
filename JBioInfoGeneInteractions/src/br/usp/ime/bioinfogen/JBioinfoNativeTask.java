package br.usp.ime.bioinfogen;

import org.joda.time.DateTime;

import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

public class JBioinfoNativeTask
    implements
        Runnable
{
    private static final long serialVersionUID = 1L;

    static {
        // Add -Djava.library.path=./native
        // Option to JVM params
        System.loadLibrary( "bioinfogen" );
    }

    private final JBioinfoTask bioinfoTask;

    public JBioinfoNativeTask(
        final ExecutionConfiguration configuration )
    {
        this.bioinfoTask = new JBioinfoTask( configuration );
    }

    public JBioinfoTask getBioinfoTask()
    {
        return bioinfoTask;
    }

    public void cancel()
    {
        // TODO: Implment cancel task
        throw new IllegalStateException( "Cancel task not implemented yet" );
    }

    @Override
    public void run()
    {
        final GeneAnalisysProgressStatus progressStatus = bioinfoTask.getProgressStatus();
        try {
            bioinfoTask.setStartDate( new DateTime() );

            final int stateId = execute( bioinfoTask.getTaskId(),
                bioinfoTask.getMatrixFileFullPath(), bioinfoTask.getGridDimDivFactor(),
                bioinfoTask.getBlockDimDivFactor(), progressStatus );

            assert stateId == bioinfoTask.getTaskState().getStateId(): "Different execution state and progressStatus state";

            bioinfoTask.setEndDate( new DateTime() );
            if( stateId == TaskState.ERROR.getStateId() ) {
                progressStatus.setState( TaskState.ERROR.getStateId() );
                System.out.println( "Undefined Error on NativeTask" );
            }
        } catch( final Throwable e ) {
            progressStatus.setState( TaskState.ERROR.getStateId() );
            System.out.println( "Error on NativeTask: " + e.getMessage() );
            e.printStackTrace();
        }
    }

    public native int execute(
        String taskId,
        String matrixFileName,
        int gridDim,
        int blockDim,
        GeneAnalisysProgressStatus status );

}
