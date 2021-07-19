package br.usp.ime.bioinfogen;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.core.task.TaskExecutor;

public class ExecutionManager
{

    private final TaskExecutor taskExecutor;
    private final Map<String,JBioinfoNativeTask> tasks;

    public ExecutionManager(
        final TaskExecutor taskExecutor )
    {
        this.taskExecutor = taskExecutor;
        this.tasks = new ConcurrentHashMap<String,JBioinfoNativeTask>();
    }

    public Map<String,JBioinfoNativeTask> getTasks()
    {
        return Collections.unmodifiableMap( tasks );
    }

    public void addTask(
        final JBioinfoNativeTask task )
    {
        tasks.put( task.getBioinfoTask().getTaskId(), task );
    }

    /**
     * Inicia execução de forma assíncrona da tarefa especificada.
     * 
     * @param taskId
     */
    public void startExecution(
        final String taskId )
    {
        final JBioinfoNativeTask task = tasks.get( taskId );
        taskExecutor.execute( task );
    }

    public void cancel(
        final String taskId )
    {
        final JBioinfoNativeTask jBioinfoNativeTask = tasks.get( taskId );
        if( jBioinfoNativeTask != null ) {
            jBioinfoNativeTask.cancel();
        }
    }
}
