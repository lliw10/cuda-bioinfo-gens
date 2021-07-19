package br.usp.ime.bioinfogen.web.controllers;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.TreeSet;

import org.joda.time.Duration;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.springframework.util.StringUtils;

import br.com.caelum.vraptor.Get;
import br.com.caelum.vraptor.Path;
import br.com.caelum.vraptor.Post;
import br.com.caelum.vraptor.Resource;
import br.com.caelum.vraptor.Result;
import br.com.caelum.vraptor.Validator;
import br.com.caelum.vraptor.interceptor.download.Download;
import br.com.caelum.vraptor.interceptor.download.FileDownload;
import br.com.caelum.vraptor.interceptor.multipart.UploadedFile;
import br.com.caelum.vraptor.ioc.RequestScoped;
import br.com.caelum.vraptor.validator.ValidationMessage;
import br.com.caelum.vraptor.validator.Validations;
import br.com.caelum.vraptor.view.Results;
import br.usp.ime.bioinfogen.AnalisysServiceException;
import br.usp.ime.bioinfogen.JBioinfoTask;
import br.usp.ime.bioinfogen.model.DataChart;
import br.usp.ime.bioinfogen.model.DataSet;
import br.usp.ime.bioinfogen.model.ExecutionConfiguration;
import br.usp.ime.bioinfogen.services.GeneAnalisysService;

import com.google.common.collect.Lists;

@RequestScoped
@Resource
public class ExecutorController
    implements
        Serializable
{
    private static final long serialVersionUID = 6888479647584877483L;

    private final Result result;
    private final Validator validator;
    private final ResourceBundle bundle;

    private final GeneAnalisysService service;

    public ExecutorController(
        final Result result,
        final Validator validador,
        final GeneAnalisysService service )
    {
        this.result = result;
        this.validator = validador;
        this.service = service;
        this.bundle = ResourceBundle.getBundle( "messages" );
    }

    @Get
    @Path( "/executor/listAll" )
    public void listAll()
    {
    }

    @Get
    @Path( "/executor/resultTable" )
    public void resultTable()
    {
        final List<JBioinfoTask> tasks = service.listAllTasks();
        result.include( "resultTaskList", tasks );
    }

    @Get
    @Path( "/executor/downloadInput/{taskId}" )
    public Download downloadInput(
        final String taskId )
    {
        FileDownload taskFileInput;
        try {
            taskFileInput = service.getInputFileDownload( taskId );
        } catch( final AnalisysServiceException e ) {
            taskFileInput = null;
            validator.add( new ValidationMessage( e.getMessage(), "error_on_execute" ) );
        }

        // Caso ocorra algum erro a requisição será redirecionada deste ponto
        validator.onErrorForwardTo( getClass() ).execute();

        return taskFileInput;
    }

    /**
     * TESTE PARA EXIBIR GRAFICO na pagina outputChart.jsp
     */
    @Get
    @Path( "/executor/outputChart" )
    public void outputChart()
    {

    }

    @Get
    @Path( "/executor/getOutputChartData/{taskId}" )
    public void getOutputChartData(
        final String taskId )
    {
        final JBioinfoTask bioinfoTask = service.getTask( taskId );

        if( bioinfoTask == null ) {
            result.use( Results.json() ).from( null, "dataChart" ).serialize();
        } else {
            final Map<Long,Long> basinSizeByAttractors = bioinfoTask.getBasinSizeByAttractors();
            if( basinSizeByAttractors == null ) {
                result.use( Results.json() ).from( null, "dataChart" ).recursive().serialize();
            } else {
                // Ordena pelo id dos atratores
                final Set<Entry<Long,Long>> entrySet = new TreeSet<Map.Entry<Long,Long>>(
                    new Comparator<Entry<Long,Long>>() {
                        @Override
                        public int compare(
                            final Entry<Long,Long> o1,
                            final Entry<Long,Long> o2 )
                        {
                            final int result = o1.getKey().compareTo( o2.getKey() );
                            return result;
                        }
                    } );
                entrySet.addAll( basinSizeByAttractors.entrySet() );

                final List<String> labels = Lists.newArrayList();

                final double maxSize = Math.pow( 2, bioinfoTask.getNumberOfGenes() );

                final int maxResults = 100;
                int i = 1;

                final DataSet dataSet = new DataSet();
                for( final Entry<Long,Long> entry : entrySet ) {
                    final Long attractorId = entry.getKey();
                    final Long basinSize = entry.getValue();

                    labels.add( attractorId.toString() );
                    dataSet.addValue( ( new Double( basinSize ) / maxSize ) * 100d );

                    i++;
                    if( i > maxResults ) {
                        break;
                    }
                }
                final DataChart dataChart = new DataChart(
                    labels,
                    Collections.singletonList( dataSet ) );

                result.use( Results.json() ).from( dataChart, "dataChart" ).recursive().serialize();
            }

        }
    }

    @Get
    @Path( "/executor/downloadOutput/{taskId}" )
    public void downloadOutput(
        final String taskId )
    {
        final JBioinfoTask bioinfoTask = service.getTask( taskId );
        final StringBuffer buffer = new StringBuffer();

        if( bioinfoTask == null ) {

            buffer.append( "<center>" );
            buffer.append( "<p>" );
            buffer.append( "Tarefa :" + taskId + " não encontrada!" );
            buffer.append( "</p>" );
            buffer.append( "</center>" );
        } else {
            final Map<Long,Long> basinSizeByAttractors = bioinfoTask.getBasinSizeByAttractors();

            buffer.append( "<br>" );
            buffer.append( "<center>" );

            buffer.append( "<table>" );
            buffer.append( "<tr>" );
            buffer.append( "<th>" );
            buffer.append( "N. de Genes" );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "Situacao" );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "Inicio" );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "Fim" );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "Duração Proc." );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "Duração Total" );
            buffer.append( "</th>" );
            buffer.append( "<th>" );
            buffer.append( "N. de atratores" );
            buffer.append( "</th>" );
            buffer.append( "</tr>" );

            final DateTimeFormatter dateFormat = DateTimeFormat.forPattern( "dd/MM/yyyy HH:mm:ss" );

            // Dados
            buffer.append( "<tr>" );
            buffer.append( "<td>" );
            buffer.append( bioinfoTask.getNumberOfGenes() );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( bundle.getString( "task.state." + bioinfoTask.getTaskState() ) );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( bioinfoTask.getStartDate().toDateTime().toString( dateFormat ) );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( bioinfoTask.getEndDate().toDateTime().toString( dateFormat ) );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( bioinfoTask.getProgressStatus().getDuration() );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( ( new Duration( bioinfoTask.getStartDate(), bioinfoTask.getEndDate() ).getMillis() ) / 1000d );
            buffer.append( "</td>" );
            buffer.append( "<td>" );
            buffer.append( basinSizeByAttractors != null ? basinSizeByAttractors.size() : 0 );
            buffer.append( "</td>" );
            buffer.append( "</tr>" );

            buffer.append( "</table>" );
            buffer.append( "</br>" );
            buffer.append( "</br>" );
            buffer.append( "</br>" );
            if( basinSizeByAttractors != null ) {
                buffer.append( "<table>" );
                buffer.append( "<tr>" );
                buffer.append( "<th>" );
                buffer.append( "Registro" );
                buffer.append( "</th>" );
                buffer.append( "<th colspan='2'>" );
                buffer.append( "Atrator" );
                buffer.append( "</th>" );

                buffer.append( "<th>" );
                buffer.append( "Tamanho da Bacia" );
                buffer.append( "</th>" );
                buffer.append( "</tr>" );

                // Ordenação inversa pela maior bacia de atração
                final Set<Entry<Long,Long>> entrySet = new TreeSet<Map.Entry<Long,Long>>(
                    Collections.reverseOrder( new Comparator<Entry<Long,Long>>() {
                        @Override
                        public int compare(
                            final Entry<Long,Long> o1,
                            final Entry<Long,Long> o2 )
                        {
                            int result = o1.getValue().compareTo( o2.getValue() );
                            if( result == 0 ) {
                                result = o1.getKey().compareTo( o2.getKey() );
                            }
                            return result;
                        }
                    } ) );
                entrySet.addAll( basinSizeByAttractors.entrySet() );

                final int maxResults = 1000;
                int i = 1;
                for( final Entry<Long,Long> entry : entrySet ) {
                    buffer.append( "<tr>" );
                    buffer.append( "<td>" );
                    buffer.append( i++ );
                    buffer.append( "</td>" );
                    buffer.append( "<td>" );
                    buffer.append( entry.getKey() );
                    buffer.append( "</td>" );
                    buffer.append( "<td>" );
                    buffer.append( Long.toBinaryString( entry.getKey() ) );
                    buffer.append( "</td>" );
                    buffer.append( "<td>" );
                    buffer.append( entry.getValue() );
                    buffer.append( "</td>" );
                    buffer.append( "</tr>" );

                    if( i > maxResults ) {
                        break;
                    }
                }
                buffer.append( "</table>" );
                buffer.append( "</br>" );
                if( i > maxResults ) {
                    buffer.append( "<p>Exibindo os " + maxResults
                        + " registros com as maiores bacias de atração</p>" );
                }
                buffer.append( "</center>" );
            }
        }
        result.include( "resultText", buffer.toString() );
        result.include( "taskId", taskId );
    }

    @Post
    @Path( "/executor/loadFile" )
    public void loadFile(
        final UploadedFile matrixFile )
    {
        validator.checking( new Validations() {
            {
                that( matrixFile != null, "error_on_execute", "error_reading_matrix_file_null" );
            }
        } );

        ExecutionConfiguration configuration = null;
        if( matrixFile != null ) {
            try {
                configuration = service.readConfiguration( matrixFile );
            } catch( final MatrixFormatException e ) {
                configuration = null;
                validator.add( new ValidationMessage( e.getMessage(), "error_on_execute" ) );
            }
        }

        // Caso ocorra algum erro a requisição será redirecionada deste ponto
        validator.onErrorForwardTo( getClass() ).execute();

        result.include( "config", configuration );

        result.forwardTo( getClass() ).execute();
    }

    @Get
    @Path( "/executor/execute" )
    public void execute()
    {
        // Load by forward or redirected actions
    }

    @Post
    @Path( "/executor/execute" )
    public void execute(
        final ExecutionConfiguration config )
    {
        final String matrixText = config.getMatrixText();
        final String matrixFileNameToValidate = config.getMatrixFileName();

        validator.checking( new Validations() {
            {
                that(
                    StringUtils.hasText( matrixText )
                        || StringUtils.hasText( matrixFileNameToValidate ), "error_on_execute",
                    "error_reading_matrix_file_null" );
                that( StringUtils.hasText( matrixText ), "error_on_execute",
                    "error_reading_matrix_file_format" );
            }
        } );

        validator.onErrorForwardTo( getClass() ).execute();

        try {
            String matrixFileName = matrixFileNameToValidate;
            if( ! StringUtils.hasText( matrixFileName ) ) {
                // O texto foi digitado diretamente na area de texto
                // Um nome padrão é dado para o arquivo
                matrixFileName = config.getTaskId();
                config.setMatrixFileName( matrixFileName );
            }
            service.execute( config );

            result.redirectTo( getClass() ).listAll();

        } catch( final MatrixFormatException e ) {
            e.printStackTrace();
            validator.checking( new Validations() {
                {
                    that( false, "error_on_execute", e.getMessage() );
                }
            } );
            validator.onErrorForwardTo( getClass() ).execute();
        } catch( final Exception e ) {
            e.printStackTrace();
            validator.checking( new Validations() {
                {
                    that( false, "error_on_execute", "error_reading_matrix_file" );

                }
            } );
            validator.onErrorForwardTo( getClass() ).execute();
        }

    }
}
