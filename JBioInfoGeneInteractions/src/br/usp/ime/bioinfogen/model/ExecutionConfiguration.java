package br.usp.ime.bioinfogen.model;

import java.io.Serializable;

public class ExecutionConfiguration
    implements
        Serializable
{
    private static final long serialVersionUID = 1L;
    public static final String TASK_ID_PREFIX = "T-";

    private final String taskId;
    private String matrixText;
    private String matrixFileName;
    private String matrixFileFullPath;
    private Integer numberOfGenes;
    private Integer gridDiv;
    private Integer blockDiv;

    public ExecutionConfiguration()
    {
        this.taskId = TASK_ID_PREFIX + Long.toString( System.currentTimeMillis() );
    }

    public ExecutionConfiguration(
        final String matrixFileFullPath,
        final String matrixFileName,
        final String matrixText,
        final Integer numberOfGenes,
        final Integer gridDiv,
        final Integer blockDiv )
    {
        this();
        this.matrixFileFullPath = matrixFileFullPath;
        this.matrixFileName = matrixFileName;
        this.matrixText = matrixText;
        this.numberOfGenes = numberOfGenes;
        this.gridDiv = gridDiv;
        this.blockDiv = blockDiv;
    }

    public String getTaskId()
    {
        return taskId;
    }

    public Integer getBlockDiv()
    {
        return blockDiv;
    }

    public void setBlockDiv(
        final Integer blockDiv )
    {
        this.blockDiv = blockDiv;
    }

    public Integer getGridDiv()
    {
        return gridDiv;
    }

    public void setGridDiv(
        final Integer gridDiv )
    {
        this.gridDiv = gridDiv;
    }

    public void setMatrixText(
        final String matrixText )
    {
        this.matrixText = matrixText;
    }

    public String getMatrixText()
    {
        return matrixText;
    }

    public void setMatrixFileName(
        final String matrixFileName )
    {
        this.matrixFileName = matrixFileName;
    }

    public String getMatrixFileName()
    {
        return matrixFileName;
    }

    public void setMatrixFileFullPath(
        final String matrixFileFullPath )
    {
        this.matrixFileFullPath = matrixFileFullPath;
    }

    public String getMatrixFileFullPath()
    {
        return matrixFileFullPath;
    }

    public void setNumberOfGenes(
        final Integer numberOfGenes )
    {
        this.numberOfGenes = numberOfGenes;
    }

    public Integer getNumberOfGenes()
    {
        return numberOfGenes;
    }
}