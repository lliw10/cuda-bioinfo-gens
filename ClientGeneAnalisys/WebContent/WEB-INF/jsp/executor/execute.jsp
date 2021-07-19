<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>

<%@ include file="../cabecalho.jsp"%>

<link href="<c:url value='/'/>/css/jquery-ui.css" rel="stylesheet"	type="text/css" />

<script type="text/javascript" src="<c:url value='/'/>/js/gdl.js"></script>
<script language="javascript" type="text/javascript" src="<c:url value='/'/>/js/interface.js"></script>
<script language="javascript" type="text/javascript" src="<c:url value='/'/>/js/jquery-ui.js"></script>
<script language="javascript" type="text/javascript" src="<c:url value='/'/>/js/dateFormat.js"></script>

<script language="javascript" type="text/javascript" src="jquery.maskedinput.js"></script>

<script type="text/javascript">
	jQuery(function($) {
		$('input.number').bind('keyup blur', function(){ 
		    var myValue = $(this).val();
		    myValue = myValue.substr(0, 4);
		    myValue = myValue.replace(/[^0-9]/g,'');		    
		    $(this).val( myValue );
		});
	});
	
	$.ajaxSetup({
		cache : false
	});
</script>

<a href="<c:url value="/executor/listAll"/>" class="handCursor"> <fmt:message key="label_link_list_all" /> </a>
<div id="init">
					
	<fieldset>
		<legend>
			<b> <fmt:message key="label_title_boolean_network_analisys" /> </b>
		</legend>
		
		<form action="<c:url value="/executor/loadFile"/>" method="post" enctype="multipart/form-data">
			<div class="row">
				<div class="label">
					<fmt:message key="label_regulation_matrix" />
				</div>
				<span class="formw">
					<input type="file" name="matrixFile" value="${matrixFile}"/> 
				</span>
				<span class="formw">
					<input name="submitMatrix" type="submit" value="<fmt:message key='label_load_matrix_button' />" />										
				</span>  
			</div>
		</form>
			
		<form action="<c:url value="/executor/execute"/>" method="post" enctype="multipart/form-data">
			
			<input type="hidden" name="config.matrixFileName"
					value="${config.matrixFileName}" maxlength=300 />
							
			<div class="row"> 
			 	<div class="label">
			 		<fmt:message key="label_file_loaded" /> ${config.matrixFileName} 
			 	</div>
			</div>
			 
			<div class="row">				
				<textarea style="width: 655px; height: 262px"
					name="config.matrixText">${config.matrixText}</textarea>
			</div>
			
			<div class="row">
				<div class="col">
					<span class="formw">	
						<fmt:message key="label_index_regulation_matrix_size" />
					</span>
					<span class="formw">
						<input class="number" type="text" name="config.numberOfGenes" value="${config.numberOfGenes}" disabled/> 
					</span>	
				<!--
					<span class="formw">	
						<fmt:message key="label_grid_div" />
					</span>
					 
					<span class="formw">
						<input class="number" type="text" name="config.gridDiv"	value="${config.gridDiv}" />							 
					</span>
				
					<span class="formw">	
						<fmt:message key="label_block_div" />
					</span>					
					<span class="formw">
						<input class="number" type="text" name="config.blockDiv" value="${config.blockDiv}" /> 
					</span>
					 -->
				</div>
			</div>
											
			<div class="row">
				<span class="formw">
				<input id="submit" name="submit" type="submit"
						value="<fmt:message key='label_execute_button' />" />
				</span>
			</div>
		</form>
		
	</fieldset>
</div>
	
<%@ include file="../rodape.jsp"%>

	