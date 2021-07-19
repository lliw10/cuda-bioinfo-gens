
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix='fn' uri="http://java.sun.com/jsp/jstl/functions"%>

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

<script type="text/javascript"
	src="<c:url value='/'/>/js/chart/Chart.min.js"></script>
<script type="text/javascript"
	src="<c:url value='/'/>/js/chart/jquery.js"></script>

<script type="text/javascript">
	$(document).ready(
			function() {
				var data = {
					labels : [ "January", "February", "March", "April", "May",
							"June", "July" ],
					datasets : [ {
						fillColor : "rgba(220,220,220,0.5)",
						strokeColor : "rgba(220,220,220,1)",
						pointColor : "rgba(220,220,220,1)",
						pointStrokeColor : "#fff",
						data : [ 65, 59, 90, 81, 56, 55, 40 ]
					}, {
						fillColor : "rgba(151,187,205,0.5)",
						strokeColor : "rgba(151,187,205,1)",
						pointColor : "rgba(151,187,205,1)",
						pointStrokeColor : "#fff",
						data : [ 28, 48, 40, 19, 96, 27, 100 ]
					} ]
				};
				
				$.ajax({
					beforeSend : function() {
					},
					url : 'outputChartData/' + 1,
					success : function(data) {
						$('#textoServer').val(data);					
					},
					error : function(txt) {
						$('#textoServer').html(txt);
					}
				});

				var options = null;
				var ctx = $("#myChart").get(0).getContext("2d");
				new Chart(ctx).Line(data, options);
				data = {
					labels : [ "0", "4", "64", "68", "384", "516", "580" ],
					datasets : [ {
						fillColor : "rgba(220,220,220,0.5)",
						strokeColor : "rgba(220,220,220,1)",
						pointColor : "rgba(220,220,220,1)",
						pointStrokeColor : "#fff",
						data : [ 7, 9, 1, 1764, 151, 7, 109 ]
					} ]
				};
				var options = null;
				var ctx = $("#myChart2").get(0).getContext("2d");
				new Chart(ctx).Bar(data, options);
			});
</script>

<p>Resultado</p>
<p id="textoServer"></p>
<canvas id="myChart" width="400" height="400"></canvas>
<canvas id="myChart2" width="400" height="400"></canvas>

<%@ include file="../rodape.jsp"%>