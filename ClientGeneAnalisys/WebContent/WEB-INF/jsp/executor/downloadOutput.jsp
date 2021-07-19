<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix='fn' uri="http://java.sun.com/jsp/jstl/functions"%>

<link href="<c:url value='/css/layout.css'/>" rel="stylesheet"
	type="text/css" />
<link href="<c:url value='/'/>/css/jquery-ui.css" rel="stylesheet"
	type="text/css" />
<script type="text/javascript" src="<c:url value='/'/>/js/gdl.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/interface.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/jquery-ui.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/dateFormat.js"></script>

<!-- Canvas JS -->
<!-- Browser support, segundo recomendado por http://www.chartjs.org/docs/ -->
<!--[if lte IE 8]><!-->
	<script src="excanvas.js"></script>
<!--<![endif]-->

<script type="text/javascript"
	src="<c:url value='/'/>/js/chart/Chart.min.js"></script>
<script type="text/javascript"
	src="<c:url value='/'/>/js/chart/jquery.js"></script>


<script type="text/javascript">

	function renderLineChart(data) {
		var dataChart = data.dataChart;
		
		if(dataChart && dataChart.labels && dataChart.dataSets && dataChart.dataSets.length > 0) {
			// Renderiza grafico de linhas
			var dataStruct = {
					labels : dataChart.labels,
					datasets : [ {
						fillColor : "rgba(220,220,220,0.5)",
						strokeColor : "rgba(220,220,220,1)",
						pointColor : "rgba(220,220,220,1)",
						pointStrokeColor : "#fff",
						data : dataChart.dataSets[0].values
					}]
				};
			
			var options = {
					   scaleOverride   : true,
					   scaleSteps      : 10,
					   scaleStepWidth  : 10,
					   scaleStartValue : 0
					};
			var options = null;
			var ctx = $("#chartLine").get(0).getContext("2d");
			new Chart(ctx).Line(dataStruct, options);
			
			//var ctx = $("#chartBar").get(0).getContext("2d");
			//new Chart(ctx).Bar(dataStruct, options);
		}
		else {
			$('#statusMessageChart').append('<h3><fmt:message key="label.download.output.msg.no.data.found" /></h3>');
		}
	}
	
	$(document).ready(
		// Carrega o grafico assincronamente
		function() {
			$.ajax({
				url : 'getOutputChartData/' + $('#taskId').val(),
			    dataType: 'json',
				beforeSend : function() {
					$('#statusMessageChart').text('');	
				},					
				success : function(data) {
					if(data.dataChart){
						renderLineChart(data);	
					}
				},
				error : function(data) {
					$('#statusMessageChart').append('<h3><fmt:message key="label.download.output.error.loading.chart" /></h3>');
				}
			});
		});
</script>

<div class="fechar_modal"> <fmt:message key="close" /> </div>

<input type="hidden" id="taskId" value="${taskId}"/>

<center>
	<h2><fmt:message key="label.download.output.result" /> <fmt:message key="list_all_title_task_id"/> ${taskId}</h2>	
	<div id="statusMessageChart"></div>
	<canvas id="chartLine" width="800" height="400"></canvas>
<!--	<canvas id="chartBar" width="800" height="400"></canvas>-->
	<div class="scrollDiv">
		${resultText}
	</div>
</center>
