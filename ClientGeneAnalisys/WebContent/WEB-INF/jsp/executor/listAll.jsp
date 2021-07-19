<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix='fn' uri="http://java.sun.com/jsp/jstl/functions"%>
<%@taglib prefix="joda" uri="http://www.joda.org/joda/time/tags" %>

<%@ include file="../cabecalho.jsp"%>

<link href="<c:url value='/'/>/css/jquery-ui.css" rel="stylesheet"
	type="text/css" />
<script type="text/javascript" src="<c:url value='/'/>/js/gdl.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/interface.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/jquery-ui.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/dateFormat.js"></script>

<script type="text/javascript">
	$.ajaxSetup({
		cache : false
	});

	var tableVisible = true;
	jQuery(function($) {

		$("#download_input_view_link").live('click', function() {
			var resultTaskId = $(this).parent().find(">:first-child");
			$.ajax({
				beforeSend : function() {
					tableVisible = false;
					$('#file_input').hide();
					$('#text_input').val("");
					//$('#fundo').show();
					$('.carregando').show();
					$('#file_input').show();					
				},
				url : 'downloadInput/' + resultTaskId.html(),
				success : function(data) {
					$('#text_input').val(data);					
					$('.carregando').hide();					
				},
				error : function(txt) {
					$('.carregando').html(txt);
				}
			});
		});
			
		$("#download_output_view_link").live('click', function() {
			var resultTaskId = $(this).parent().find(">:first-child");
			$.ajax({
				beforeSend : function() {
					tableVisible = false;
					$('#file_output').html("");
					//$('#fundo').show();
					$('.carregando').show();
					$('#file_output').show();
				},
				url : 'downloadOutput/' + resultTaskId.html(),
				success : function(data) {
					$('#file_output').html(data);
					$('.carregando').hide();					
				},
				error : function(txt) {
					$('.carregando').html(txt);
				}
			});
		});
		
		$('.fechar_modal').live('click', function(){
			$('#file_input').hide();
			$('#file_output').hide();
			//$('#fundo').hide();
			tableVisible = true;
		});
	});

	var reloadData = 0; // store timer
	var reloadTime = 100;
	var maxReloadTime = 10000; // 10 segundos
	
	$(document).ready(function() {
        // load data on page load and sets timeout to reload again
        $('#reloadAuto').attr("checked",true);
		$('#reloadAuto').click(function() {
	        if ($(this).is(':checked')) {
	        	loadData();	            
	        }
	        else {
	        	window.clearTimeout(reloadData);
	        }
	    });
        loadData();
    });

    function loadData() {
    	if(tableVisible) {
    		reloadTime = reloadTime >= maxReloadTime ? maxReloadTime : reloadTime * 2    
        	$.ajax({
    			beforeSend : function() {
    				$('.carregando').toggle();
    			},
    			url : 'resultTable',
    			success : function(data) {
    				$('.carregando').toggle();
    				$('#resultTable').html(data);
    				if (reloadData != 0) {
    	            	window.clearTimeout(reloadData);
    	            }
    				var reloadAuto = $('#reloadAuto').attr("checked");
    				if(reloadAuto) {
    					// Executa a funcao loadData apos reloadTime 
    					reloadData = window.setTimeout(loadData, reloadTime);	
    				}
    			},
    			error : function(data) {
    				$('.carregando').html(data);
    			}
    		});         
    	}
    }

</script>

<div style="text-align: right;">
	<div class="carregando"></div>
	<input style="text-align: right" type="checkbox" id="reloadAuto" name="reloadAuto"><fmt:message key="list_all_reload_auto"/>	
</div>

<div id="resultTable">
<!-- resultTable.jsp sera carregado aqui -->
</div>

<div class="clear"></div>

<div id="file_output" class="modal">
	<!-- downloadOutput.jsp sera carregado aqui -->
</div>

<div id="file_input" class="modal">
	<div class="fechar_modal"> <fmt:message key="close" /> </div>
	<center>
	<br>
	<h2>Entrada - Matriz de regulação gênica</h2>	
	</center>
	<center>
		<br>	
		<br>
		<textarea id="text_input" rows="20" cols="50" >
			${resultText}
		</textarea>
	</center>	
</div>

<%@ include file="../rodape.jsp"%>
