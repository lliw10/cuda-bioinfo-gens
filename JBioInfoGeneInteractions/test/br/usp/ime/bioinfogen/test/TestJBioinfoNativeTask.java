package br.usp.ime.bioinfogen.test;

import java.util.Map;
import java.util.Map.Entry;

import org.joda.time.Duration;
import org.junit.Assert;
import org.junit.Test;

import br.usp.ime.bioinfogen.GeneAnalisysProgressStatus;
import br.usp.ime.bioinfogen.JBioinfoNativeTask;
import br.usp.ime.bioinfogen.JBioinfoTask;
import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

public class TestJBioinfoNativeTask
{
    @Test
    public void test1()
    {
        testExecuteTask();
    }

    public void testExecuteTask()
    {
        final String _matrixFileName = "/home/william/Documentos/work/sources/cuda/BioinfoGeneInteractions/mat_20.txt";

        final long startmillis = System.currentTimeMillis();

        final ExecutionConfiguration configuration = new ExecutionConfiguration(
            _matrixFileName,
            _matrixFileName,
            null,
            20,
            2,
            2 );
        final JBioinfoNativeTask task = new JBioinfoNativeTask( configuration );
        final Thread thread = new Thread( task, task.getBioinfoTask().getTaskId() );
        thread.setDaemon( true );
        thread.start();
        try {
            thread.join();
        } catch( final InterruptedException e ) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        final JBioinfoTask bioinfoTask = task.getBioinfoTask();
        final GeneAnalisysProgressStatus progressStatus = bioinfoTask.getProgressStatus();
        final Map<Long,Long> basinSizeByAttractors = bioinfoTask.getBasinSizeByAttractors();

        System.out.println();
        System.out.println( "----------------- JAVA RESULT --------------------\n" );
        System.out.println( "Number of GPUs: " + progressStatus.getnGpus() );
        System.out.println( "Grid Dim: " + progressStatus.getBlockDim() );
        System.out.println( "Block Dim: " + progressStatus.getThreadDim() );

        for( int i = 0; i < progressStatus.getnGpus(); i++ ) {
            System.out.println( "ProgressIndicator GPU[" + i + "] "
                + progressStatus.getProgressIndicator( i ) + "%" );
        }
        System.out.println( "Start: " + bioinfoTask.getStartDate() );
        System.out.println( "End: " + bioinfoTask.getEndDate() );
        System.out.println( "C Processing Time:" + progressStatus.getDuration() );
        System.out.println( "Java Processing Date Time (s): "
            + ( new Duration( bioinfoTask.getStartDate(), bioinfoTask.getEndDate() ).getMillis() )
            / 1000d );
        System.out.println( "Java Processing Millis Time (s): "
            + ( System.currentTimeMillis() - startmillis ) / 1000d );

        System.out.println( "Processing State: " + bioinfoTask.getTaskState() );
        if( basinSizeByAttractors != null ) {
            System.out.println( "Number of attractors: " + basinSizeByAttractors.size() );

            for( final Entry<Long,Long> entry : basinSizeByAttractors.entrySet() ) {
                System.out.println( "Attractor: " + entry.getKey() + " Basin Size: "
                    + entry.getValue() );
            }

            Assert.assertTrue( basinSizeByAttractors.size() > 0 );

            System.out.println( "Result finish" );
        }
    }

    public static void main(
        final String args[] )
    {
        final TestJBioinfoNativeTask testJBioinfoNativeTask = new TestJBioinfoNativeTask();
        for( int i = 0; i < 1; i++ ) {
            System.out.println( "Free memory pre: " + Runtime.getRuntime().freeMemory() );
            testJBioinfoNativeTask.testExecuteTask();
            System.out.println( "Free memory post: " + Runtime.getRuntime().freeMemory() );
            System.out.println();
        }

    }
}
