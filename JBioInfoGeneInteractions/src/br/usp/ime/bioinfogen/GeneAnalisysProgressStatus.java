package br.usp.ime.bioinfogen;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Status of execution C/C++ call. The fields of this class are controlled by
 * C/C++ code, possibly changed by another (native) thread.
 */
public class GeneAnalisysProgressStatus
    implements
        Serializable
{
    private static final long serialVersionUID = 1L;

    /**
     * Number of GPUs to use as co-processor.
     */
    private int nGpus;

    /**
     * Number of blocks used in execution.
     */
    private int blockDim;

    /**
     * Number of threads used in execution.
     */
    private int threadDim;

    /**
     * State of Progress. @see{TaskState} .
     */
    private int state;

    /**
     * Start index to i-GPU process data.
     */
    private int[] startIdx;

    /**
     * End index to i-GPU process data.
     */
    private int[] endIdx;

    /**
     * Progress indicator to each GPU processor.
     */
    private double[] progressIndicator;

    /**
     * Number of attractors founded. This field is setted only when process is
     * finished.
     */
    private int numberOfAttractors;

    /**
     * Array of all Attractor States.
     */
    private long[] attractorStates;

    /**
     * Array of size of each Attractor State.
     */
    private long[] attractorBasinSize;

    /**
     * Progress time execution in seconds.
     */
    private double duration;

    /**
     * Convenience map to access attractors and sizes. This map is transient.
     */
    private transient Map<Long,Long> basinSizeByAttractors;

    public GeneAnalisysProgressStatus()
    {
        this.state = TaskState.NOT_SUBMITTED.getStateId();
        this.numberOfAttractors = 0;
    }

    public void buildAttractorStateArrays(
        final int numberOfAttractors )
    {
        this.numberOfAttractors = numberOfAttractors;
        this.attractorStates = new long[ this.numberOfAttractors ];
        this.attractorBasinSize = new long[ this.numberOfAttractors ];
    }

    public long[] getAttractorStates()
    {
        return attractorStates;
    }

    public long[] getAttractorBasinSize()
    {
        return attractorBasinSize;
    }

    public int getBlockDim()
    {
        return blockDim;
    }

    public int getThreadDim()
    {
        return threadDim;
    }

    public int getStartIdx(
        final int gpuIdx )
    {
        return startIdx[ gpuIdx ];
    }

    public int getEndIdx(
        final int gpuIdx )
    {
        return endIdx[ gpuIdx ];
    }

    public int getnGpus()
    {
        return nGpus;
    }

    public double getDuration()
    {
        return duration;
    }

    public double getProgressIndicator(
        final int gpuIdx )
    {
        if( progressIndicator != null && gpuIdx < nGpus ) {
            return progressIndicator[ gpuIdx ];
        }
        return 0d;
    }

    protected void setState(
        final int state )
    {
        this.state = state;
    }

    /**
     * Define progress indicator value. if Multiple threads
     * 
     * @param deviceCount
     * @param deviceId
     * @param progressIndicatorValue
     */
    public void setProgressIndicator(
        final int deviceCount,
        final int deviceId,
        final double progressIndicatorValue )
    {
        if( this.progressIndicator == null ) {
            synchronized( this ) {
                if( this.progressIndicator == null ) {
                    this.progressIndicator = new double[ deviceCount ];
                }
            }
        }
        this.progressIndicator[ deviceId ] = progressIndicatorValue;
    }

    public TaskState getTaskState()
    {
        return TaskState.valueOf( state );
    }

    public Map<Long,Long> getBasinSizeByAttractors()
    {
        if( basinSizeByAttractors == null && getTaskState().finalized() ) {
            basinSizeByAttractors = new HashMap<Long,Long>();
            for( int i = 0; i < attractorStates.length; i++ ) {
                final long attractorId = attractorStates[ i ];
                final long basinSize = attractorBasinSize[ i ];
                basinSizeByAttractors.put( attractorId, basinSize );
            }
        }
        return basinSizeByAttractors;
    }
}
