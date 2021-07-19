package br.usp.ime.bioinfogen;

import java.util.List;

import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

/**
 * Fachada para controle da execução de tarefas de background. Disponibiliza
 * também
 * 
 * @author william
 */
public interface GeneAnalisysExecutorFacade
{

    void execute(
        ExecutionConfiguration config )
        throws AnalisysServiceException;

    void requestCancel(
        String taskId );

    List<JBioinfoTask> getTasks();

    JBioinfoTask getTask(
        String taskId );

    byte[] getInputFile(
        final String taskId )
        throws AnalisysServiceException;
}
