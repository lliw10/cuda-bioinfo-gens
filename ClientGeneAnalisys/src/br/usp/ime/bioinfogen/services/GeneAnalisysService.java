package br.usp.ime.bioinfogen.services;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.util.StringUtils;

import br.com.caelum.vraptor.interceptor.download.FileDownload;
import br.com.caelum.vraptor.interceptor.multipart.UploadedFile;
import br.com.caelum.vraptor.ioc.Component;
import br.com.caelum.vraptor.ioc.RequestScoped;
import br.usp.ime.bioinfogen.AnalisysServiceException;
import br.usp.ime.bioinfogen.JBioinfoTask;
import br.usp.ime.bioinfogen.RemoteGeneAnalisysFacade;
import br.usp.ime.bioinfogen.model.ExecutionConfiguration;
import br.usp.ime.bioinfogen.web.controllers.MatrixFormatException;

@RequestScoped
@Component
public class GeneAnalisysService
    implements
        Serializable
{
    private static final long serialVersionUID = - 7152815489716947464L;
    private static final String LINE_SEPARATOR_COMMA = ";";
    private static final String LINE_SEPARATOR_SPACE = " ";
    private static final String LINE_SEPARATOR_DOUBLE_SPACE = "  ";
    private static final String LINE_COMMENT = "#";
    private static final String LINE_BREAK = "\n";

    private final RemoteGeneAnalisysFacade caller;

    @Autowired
    public GeneAnalisysService(
        @Qualifier( "remote.gene.analisys.facade" )
        final RemoteGeneAnalisysFacade caller )
    {
        this.caller = caller;

    }

    public byte[] convertFileToByteArray(
        final UploadedFile matrixFile )
        throws IOException
    {
        return IOUtils.toByteArray( matrixFile.getFile() );
    }

    public FileDownload convertBytesToFileDownload(
        final String nomeArquivo,
        final byte[] conteudoBytes )
    {
        if( nomeArquivo != null && conteudoBytes != null ) {
            final String contentType = "text/plain";
            final StringBuilder fileName = new StringBuilder();

            fileName.append( nomeArquivo );
            fileName.append( ".txt" );
            final File file = new File( fileName.toString() );
            FileDownload fileDownloadCarregado = null;
            try {
                FileUtils.writeByteArrayToFile( file, conteudoBytes );
                fileDownloadCarregado = new FileDownload( file, contentType, fileName.toString() );
            } catch( final IOException e ) {
                fileDownloadCarregado = null;
            }
            return fileDownloadCarregado;
        } else {
            return null;
        }
    }

    private void persist(
        final String taskId,
        final String matrixFileName,
        final String matrixData,
        final int blockDimDiv,
        final int threadDimDiv )
        throws IOException
    {
        final String tmpFolder = "/tmp/bioinfogen/";
        final File folder = new File( tmpFolder );
        if( ! folder.exists() ) {
            folder.mkdirs();
        }
        final File file = new File( tmpFolder + taskId + "-" + matrixFileName );
        file.createNewFile();
        final FileOutputStream fout = new FileOutputStream( file );
        fout.write( matrixData.getBytes() );
        fout.close();
    }

    public void execute(
        final ExecutionConfiguration configuration )
        throws AnalisysServiceException,
            MatrixFormatException
    {
        final ExecutionConfiguration config = buildConfiguration(
            configuration.getMatrixFileName(), configuration.getMatrixText() );

        caller.execute( config );
    }

    public List<JBioinfoTask> listAllTasks()
    {
        return caller.listAllTasks();
    }

    public JBioinfoTask getTask(
        final String taskId )
    {
        return caller.getTask( taskId );
    }

    public ExecutionConfiguration readConfiguration(
        final UploadedFile matrixFile )
        throws MatrixFormatException
    {
        if( matrixFile == null ) {
            throw new IllegalStateException( "Null matrix file" );
        }
        try {
            final BufferedReader reader = new BufferedReader( new InputStreamReader(
                matrixFile.getFile() ) );

            final StringBuffer buffer = new StringBuffer();

            String line = reader.readLine();
            while( line != null ) {
                buffer.append( line );
                buffer.append( LINE_BREAK );
                line = reader.readLine();
            }
            return buildConfiguration( matrixFile.getFileName(), buffer.toString() );

        } catch( final IOException e ) {
            throw new MatrixFormatException( "error_reading_matrix_file_null" );
        }
    }

    private ExecutionConfiguration buildConfiguration(
        final String fileName,
        final String text )
        throws MatrixFormatException
    {
        if( ! StringUtils.hasText( fileName ) ) {
            throw new IllegalStateException( "Empty file name" );
        }

        if( ! StringUtils.hasText( text ) ) {
            throw new IllegalStateException( "Empty file name" );
        }

        final StringBuffer buffer = new StringBuffer();
        final String lines[] = text.split( LINE_BREAK );
        boolean firstLine = true;
        int numberOfGenes = 0;
        if( lines.length > 0 ) {
            for( final String lineSplited : lines ) {
                String line = lineSplited.trim();
                if( ! line.isEmpty() && ! line.startsWith( LINE_COMMENT ) ) {
                    if( firstLine ) {
                        try {
                            numberOfGenes = Integer.parseInt( line );
                        } catch( final NumberFormatException e ) {
                            numberOfGenes = 0;
                            throw new MatrixFormatException( "error_reading_matrix_file_format" );
                        }
                        firstLine = false;
                    } else {

                        while( line.contains( LINE_SEPARATOR_DOUBLE_SPACE ) ) {
                            line = line.replace( LINE_SEPARATOR_DOUBLE_SPACE, LINE_SEPARATOR_SPACE );
                        }

                        final String[] split = line.split( LINE_SEPARATOR_SPACE );
                        if( split == null || split.length != numberOfGenes ) {
                            throw new MatrixFormatException( "error_reading_matrix_file_format" );
                        }
                    }

                    buffer.append( lineSplited );
                    buffer.append( LINE_BREAK );
                }
            }
        }
        return new ExecutionConfiguration(
            fileName,
            fileName,
            buffer.toString(),
            numberOfGenes,
            getGridDiv(),
            getBlockDiv() );
    }

    public int getGridDiv()
    {
        // TODO: get config by some external param
        return 2;
    }

    public int getBlockDiv()
    {
        // TODO: get config by some external param
        return 2;
    }

    public FileDownload getInputFileDownload(
        final String taskId )
        throws AnalisysServiceException
    {
        final byte[] inputFile = caller.getInputFile( taskId );
        final FileDownload fileDownload = this.convertBytesToFileDownload( "matrix-" + taskId,
            inputFile );
        return fileDownload;
    }
}