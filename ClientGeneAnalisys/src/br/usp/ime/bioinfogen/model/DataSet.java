package br.usp.ime.bioinfogen.model;

import java.util.ArrayList;
import java.util.List;

public class DataSet
{
    private final List<Double> values;

    public DataSet()
    {
        this.values = new ArrayList<Double>();
    }

    public DataSet(
        final List<Double> values )
    {
        super();
        this.values = values;
    }

    public List<Double> getValues()
    {
        return values;
    }

    public void addValue(
        final Double value )
    {
        values.add( value );
    }
}