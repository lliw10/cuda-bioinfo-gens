package br.usp.ime.bioinfogen;

import java.util.HashMap;
import java.util.Map;

public enum TaskState
{
    NOT_SUBMITTED( 0 ),
    RUNNING( 1 ),
    FINALIZED( 2 ),
    CANCELLED( 3 ),
    ERROR( - 1 );

    private Integer stateId;

    private static final Map<Integer,TaskState> states;
    static {
        states = new HashMap<Integer,TaskState>();
        states.put( NOT_SUBMITTED.getStateId(), NOT_SUBMITTED );
        states.put( RUNNING.getStateId(), RUNNING );
        states.put( FINALIZED.getStateId(), FINALIZED );
        states.put( CANCELLED.getStateId(), CANCELLED );
        states.put( ERROR.getStateId(), ERROR );
    }

    private TaskState(
        final int stateId )
    {
        this.stateId = stateId;
    }

    public Integer getStateId()
    {
        return stateId;
    }

    public static TaskState valueOf(
        final int stateId )
    {
        return states.get( stateId );
    }

    public boolean finalized()
    {
        return this == FINALIZED;
    }
}
