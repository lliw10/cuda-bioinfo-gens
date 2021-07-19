<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix='fn' uri="http://java.sun.com/jsp/jstl/functions"%>
<%@taglib prefix="joda" uri="http://www.joda.org/joda/time/tags" %>

<link href="<c:url value='/'/>/css/jquery-ui.css" rel="stylesheet"
	type="text/css" />
<script type="text/javascript" src="<c:url value='/'/>/js/gdl.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/interface.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/jquery-ui.js"></script>
<script language="javascript" type="text/javascript"
	src="<c:url value='/'/>/js/dateFormat.js"></script>

<div class="row">	
	<c:choose>
		<c:when test="${fn:length(resultTaskList) == 1}">
			<fmt:message key="list_all_task_count">
				<fmt:param value="${fn:length(resultTaskList)}" />
			</fmt:message>
		</c:when>
		<c:otherwise>
			<fmt:message key="list_all_task_count_plural">
				<fmt:param value="${fn:length(resultTaskList)}" />
			</fmt:message>		
		</c:otherwise>
	</c:choose>
	-
	<span>
		<a href="<c:url value="/executor/execute"/>" class="handCursor"> 
			<fmt:message key="list_all_link_new_task" /> 
		</a>
	</span>
</div>

<div class="column" style="width: 100%">
	<div class="row">
		<table class="tabela">
			<tr>
				<th></th>
				<th><b><fmt:message key="list_all_title_task_id" /></b></th>
				<th><b><fmt:message key="list_all_title_task_number_of_genes" /></b></th>
				<th><b><fmt:message key="list_all_title_task_state" /></b></th>
				<th><b><fmt:message key="list_all_title_task_progressIndicator" /></b></th>				
				<th><b><fmt:message key="list_all_title_task_start_date" /></b></th>
				<th><b><fmt:message key="list_all_title_task_end_date" /></b></th>
				<th><b><fmt:message key="list_all_title_task_duration" /></b></th>
				<th><b><fmt:message key="list_all_title_task_input_file" /></b></th>
				<th><b><fmt:message key="list_all_title_task_output_file" /></b></th>
				<th></th>
			</tr>
			<c:forEach var="result" items="${resultTaskList}" varStatus="rowCounter">
				<c:choose>
					<c:when test="${rowCounter.count % 2 == 0}">
						<c:set var="rowStyle" scope="page" value="odd" />
					</c:when>
					<c:otherwise>
						<c:set var="rowStyle" scope="page" value="even" />
					</c:otherwise>
				</c:choose>
										
				<tr class="${rowStyle}">
					<td>${rowCounter.count}</td>
					<td>${result.taskId}</td>
					<td>${result.getNumberOfGenes()}</td>
					<td><fmt:message key="task.state.${result.taskState}" /></td>
					<td>
					<c:forEach var="gpuId" begin="1" end="${result.progressStatus.nGpus}">
						<c:set var="progress" value='${result.progressStatus.getProgressIndicator(gpuId-1)}'/>
						Thread GPU <c:out value="${gpuId} "/> (<c:out value="${progress}"/>%) 
						<br>   
					</c:forEach>
					<td><joda:format value="${result.startDate}" style="MM" /></td>
					<td><joda:format value="${result.endDate}" style="MM" /></td>					  
					<td>${result.progressStatus.duration}</td>
					<td>
						<span style="display: none">${result.taskId}</span> <a id="download_input_view_link"  class="handCursor"> <fmt:message
						key="list_all_link_task_input_file" /> </a>						 
					</td>
					<td><span style="display: none">${result.taskId}</span> <a id="download_output_view_link" class="handCursor"> <fmt:message
						key="list_all_link_task_ouput_file" /> </a></td>
				</tr>
			</c:forEach>
		</table>
	</div>	
</div>
