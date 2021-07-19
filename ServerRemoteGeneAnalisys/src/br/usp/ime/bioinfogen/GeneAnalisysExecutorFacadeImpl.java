package br.usp.ime.bioinfogen;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import br.usp.ime.bioinfogen.model.ExecutionConfiguration;

public class GeneAnalisysExecutorFacadeImpl
    implements
        GeneAnalisysExecutorFacade
{
    private final Logger logger = Logger.getLogger( getClass() );

    class FileManager
    {
        private final String workFolder;

        public FileManager()
        {
            this.workFolder = "/tmp/bioinfogen/";
        }

        public String createFile(
            final String fileName,
            final String matrixData )
            throws IOException
        {
            final File folder = new File( workFolder );
            if( ! folder.exists() ) {
                folder.mkdirs();
            }
            final File file = new File( workFolder + fileName );
            file.createNewFile();

            final FileOutputStream fout = new FileOutputStream( file );
            fout.write( matrixData.replace( "\n", "" ).trim().getBytes() );
            fout.close();

            return file.getPath();
        }

        public byte[] getFileBytes(
            final String fileName )
            throws IOException
        {
            final File file = new File( fileName );
            if( ! file.exists() ) {
                return null;
            }
            final long size = file.length();
            final byte[] bytes = new byte[ (int) size ];
            final FileInputStream input = new FileInputStream( fileName );

            input.read( bytes );

            return bytes;
        }
    }

    private final ExecutionManager manager;
    private final FileManager fileManager;

    public GeneAnalisysExecutorFacadeImpl(
        final ExecutionManager manager )
    {
        this.manager = manager;
        this.fileManager = new FileManager();
        logger.info( "Remote " + getClass().getName() + " started" );
    }

    @Override
    public void execute(
        final ExecutionConfiguration configuration )
        throws AnalisysServiceException
    {
        final String taskId = configuration.getTaskId();
        final String fileName = configuration.getMatrixFileName();
        final String matrixData = configuration.getMatrixText();
        try {
            logger.info( "Executing task " + configuration.getTaskId() + " File: "
                + configuration.getMatrixFileName() + " MatrixData: \n"
                + configuration.getMatrixText() );

            String matrixFilePath;
            matrixFilePath = fileManager.createFile( taskId + "-" + fileName, matrixData );
            configuration.setMatrixFileFullPath( matrixFilePath );
            logger.info( "Create input file:" + matrixFilePath );

            manager.addTask( new JBioinfoNativeTask( configuration ) );

            logger.info( "Starting task: " + taskId );

            manager.startExecution( taskId );

        } catch( final IOException e ) {
            throw new AnalisysServiceException( e );
        }

    }

    @Override
    public void requestCancel(
        final String taskId )
    {
        manager.cancel( taskId );
    }

    @Override
    public List<JBioinfoTask> getTasks()
    {
        final Map<String,JBioinfoNativeTask> tasks = manager.getTasks();
        final List<JBioinfoTask> taskList = new ArrayList<JBioinfoTask>();
        if( tasks != null ) {
            for( final JBioinfoNativeTask task : tasks.values() ) {
                taskList.add( task.getBioinfoTask() );
            }
        }
        Collections.sort( taskList, new Comparator<JBioinfoTask>() {
            @Override
            public int compare(
                final JBioinfoTask o1,
                final JBioinfoTask o2 )
            {
                int comp = o1.getStartDate().compareTo( o2.getStartDate() );
                if( comp == 0 ) {
                    comp = o1.getTaskId().compareTo( o2.getTaskId() );
                }
                return comp;
            }
        } );

        return taskList;
    }

    @Override
    public JBioinfoTask getTask(
        final String taskId )
    {
        final Map<String,JBioinfoNativeTask> tasks = manager.getTasks();
        if( tasks != null ) {
            final JBioinfoNativeTask nativeTask = tasks.get( taskId );
            if( nativeTask != null ) {
                return nativeTask.getBioinfoTask();
            }
        }
        return null;
    }

    @Override
    public byte[] getInputFile(
        final String taskId )
        throws AnalisysServiceException
    {
        try {
            final JBioinfoTask task = getTask( taskId );

            return fileManager.getFileBytes( task.getMatrixFileFullPath() );

        } catch( final Exception e ) {
            throw new AnalisysServiceException( e );
            // TODO: handle exception
        }
    }
}
