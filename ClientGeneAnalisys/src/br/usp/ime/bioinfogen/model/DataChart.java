package br.usp.ime.bioinfogen.model;

import java.util.List;

public class DataChart
{
    private final List<String> labels;
    private final List<DataSet> dataSets;

    public DataChart(
        final List<String> labels,
        final List<DataSet> dataSets )
    {
        this.labels = labels;
        this.dataSets = dataSets;
    }

    public List<String> getLabels()
    {
        return labels;
    }

    public List<DataSet> getDataSets()
    {
        return dataSets;
    }
}
