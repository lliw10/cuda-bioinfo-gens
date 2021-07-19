package br.usp.ime.bioinfogen;

import java.io.Serializable;
import java.util.Map;

import org.joda.time.DateTime;

import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

public class JBioinfoTask
    implements
        Serializable
{
    private static final long serialVersionUID = 1L;

    /**
     * Input to native method.
     */
    protected ExecutionConfiguration configuration;

    /**
     * Date time execution params
     */
    protected DateTime startDate;
    protected DateTime endDate;

    /**
     * Progress status and result.
     */
    protected final GeneAnalisysProgressStatus progressStatus;

    public JBioinfoTask(
        final ExecutionConfiguration configuration )
    {
        this.configuration = configuration;
        this.progressStatus = new GeneAnalisysProgressStatus();
    }

    public String getTaskId()
    {
        return configuration.getTaskId();
    }

    public Integer getNumberOfGenes()
    {
        return configuration.getNumberOfGenes();
    }

    public String getMatrixFileName()
    {
        return configuration.getMatrixFileName();
    }

    public String getMatrixFileFullPath()
    {
        return configuration.getMatrixFileFullPath();
    }

    public int getGridDimDivFactor()
    {
        return configuration.getGridDiv();
    }

    public int getBlockDimDivFactor()
    {
        return configuration.getBlockDiv();
    }

    public DateTime getStartDate()
    {
        return startDate;
    }

    public void setStartDate(
        final DateTime startDate )
    {
        this.startDate = startDate;
    }

    public DateTime getEndDate()
    {
        return endDate;
    }

    public void setEndDate(
        final DateTime endDate )
    {

        this.endDate = endDate;
    }

    public TaskState getTaskState()
    {
        return progressStatus.getTaskState();
    }

    public Map<Long,Long> getBasinSizeByAttractors()
    {
        return progressStatus.getBasinSizeByAttractors();
    }

    public GeneAnalisysProgressStatus getProgressStatus()
    {
        return progressStatus;
    }

}