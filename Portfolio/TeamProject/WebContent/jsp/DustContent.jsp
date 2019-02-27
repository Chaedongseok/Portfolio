<%@page import="Model.vo.RegionDataVO"%>
<%@page import="java.util.ArrayList"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="java.util.*, java.text.*"%>

<%
	Calendar cal = Calendar.getInstance();
	int year = cal.get(Calendar.YEAR);
	int month = cal.get(Calendar.MONTH);
	int day = cal.get(Calendar.DAY_OF_MONTH);
	int hour = cal.get(Calendar.HOUR);
	int minute = cal.get(Calendar.MINUTE);
	int second = cal.get(Calendar.SECOND);
	int millisecond = cal.get(Calendar.MILLISECOND);
%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
	<%
		if (request.getParameter("status_num") != null) {
	%>
	<%
		float[][] list = (float[][]) session.getAttribute("regiondata");
			float[] region = list[(Integer.parseInt(request.getParameter("status_num")) - 1)];
			String[] regionname_arr = { "강남", "강동", "강북", "관악", "광진", "구로", "금천", "노원", "동대문", "동자", "마포", "서대문",
					"성북", "송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑" };
			String regionname = regionname_arr[(Integer.parseInt(request.getParameter("status_num")) - 1)];
	%>
	<h3><%=regionname%>구
	</h3>
	<h1><%=region[hour]%></h1>

	<div class="monitor">
		<div class="monitor_t" id="date_1"><%=year%>년
			<%=(month + 1)%>월
			<%=(day + 1)%>일
			<%=hour%>시
		</div>
		<hr align="center" style="border: solid 1px #333; width: 50%;">
		<div class="monitor_m">
			<table width="100%" cellspacing="0" cellpadding="0"
				summary="선택된 지역별 측정소의 초미세먼지, 미세먼지, 오존, 이산화질소, 일산화탄소, 아황산가스의 정보를 테이블로 보여줍니다.">
				<tbody>
					<tr>
						<th scope="row" style="text-align: left; padding-left: 5px;">초미세먼지</th>
						<td id="pm25_1">43 µg/m³</td>
					</tr>
					<tr>
						<th scope="row" style="text-align: left; padding-left: 5px;">오존</th>
						<td id="o3_1">0.036 ppm</td>
					</tr>
					<tr>
						<th scope="row" style="text-align: left; padding-left: 5px;">이산화질소</th>
						<td id="no2_1">0.018 ppm</td>
					</tr>
					<tr>
						<th scope="row" style="text-align: left; padding-left: 5px;">일산화탄소</th>
						<td id="co_1">0.8 ppm</td>
					</tr>
					<tr class="bdr_none">
						<th scope="row" style="text-align: left; padding-left: 5px;">아황산가스</th>
						<td id="so2_1">0.005 ppm</td>
					</tr>
				</tbody>
			</table>
		</div>
	</div>
	<div class="notes">
		<div class="notes_t">
			<h4>미세먼지 단계</h4>
		</div>
		<div class="notes_m">
			<table width="100%" cellspacing="0" cellpadding="0"
				summary="오존의 대한 범례와 지수범위를 색상과 수치로 나타냅니다. 파란색은 좋음, 연두색은 보통, 노랑색은 나쁨, 빨간색은 매우나쁨 , 흰색은 점검중을 의미합니다.">
				<thead>
					<tr>
						<th scope="col" style ="text-align: center;">단계</th>
						<td>기준값</td>
					</tr>
				</thead>
				<tbody>
					<tr>
						<th scope="row" class="num_01"
							style="text-align: left; padding-left: 5px;">① 좋음</th>
						<td>0~30</td>
					</tr>
					<tr>
						<th scope="row" class="num_02"
							style="text-align: left; padding-left: 5px;">② 보통</th>
						<td>31~50</td>
					</tr>
					<tr>
						<th scope="row" class="num_03"
							style="text-align: left; padding-left: 5px;">③ 나쁨</th>
						<td>51~100</td>
					</tr>
					<tr>
						<th scope="row" class="num_04"
							style="text-align: left; padding-left: 5px;">④ 매우나쁨</th>
						<td>101~</td>
					</tr>
				</tbody>
			</table>
		</div>
		<div class="notes_b"></div>
	</div>
	<%} %>
</body>
</html>