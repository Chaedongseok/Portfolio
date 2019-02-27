<%@page import="java.util.Arrays"%>
<%@page import="java.util.Calendar"%>
<%@page import="Model.vo.RecommendVO"%>
<%@page import="java.util.ArrayList"%>
<%@page import="Model.vo.CustomInfoVO"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<head>
<meta charset="UTF-8">

<link rel="stylesheet"
	href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">
<link rel="stylesheet"
	href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap-theme.min.css">
<link rel="stylesheet" href="/TeamProject/jsp/css/DustTotal.css">
<link rel="stylesheet" href="/TeamProject/jsp/css/FooterTotal.css">
<script src="//code.jquery.com/jquery-1.11.0.min.js"></script>
<script
	src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
<script
	src="//cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
<title>내일미세</title>
<style type="text/css">
body {
	background-color: #fafafa;
}

.survey-contents-inner {
	position: relative;
	margin: 0 auto;
	padding-bottom: 60px;
	width: 940px;
	background-color: #fff;
	box-shadow: 0px 2px 3px 0 rgba(0, 0, 0, 0.3);
	text-align: center;
}

.page {
	position: relative;
	margin: 0 auto;
	width: 940px;
	text-align: center;
}

.start-header {
	position: relative;
	height: 286px;
	background-image: url(/images/bg-survey-pattern.png);
}

h2 {
	padding-top: 50px;
	font-size: 40px;
	font-weight: 400;
	text-align: center;
}

.colored {
	color: #f1574b;
}

.Mainicon {
	margin-top: 25px;
	height: 160px;
}

p {
	padding: 30px 0 30px;
	line-height: 25px;
	font-size: 16px;
	color: #999;
}

.btn {
	cursor: pointer;
	width: 418px;
	
}

.btn-dark {
	background-color: #333;
	border: none;
	color: #fff;
}

.btn-radius {
	border-radius: 25px;
}

ul, ol, li {
	list-style: none;
	display: block;
	list-style-type: disc;
	margin-block-start: 1em;
	margin-block-end: 1em;
	margin-inline-start: 0px;
	margin-inline-end: 0px;
	padding-inline-start: 0px;
}

.survey-test-header {
	height: 91px;
	box-sizing: border-box;
}

.step-list-wrap {
	display: table;
	margin-right: auto;
	margin-left: auto;
	padding-left: 10px;
	width: 450px;
}

.step-list-wrap li {
	position: relative;
	height: 90px;
	font-size: 18px;
	font-weight: 700;
	text-align: center;
	color: #bdbdbd;
	vertical-align: middle;
	float: left;
	margin-right: 50px;
}

.glyphicon {
	padding-left: 50px;
	color: #C3D69B;
}

.survey-question-contents {
	padding-top: 36px;
	margin: 0 auto;
	width: 670px;
}

.question-answer-wrap .label-question {
	line-height: 20px;
}

.label-checkbox, .label-radio {
	position: relative;
	display: inline-block;
	line-height: 24px;
	min-height: 24px;
	font-size: 20px;
	color: #333;
}

.label-checkbox .input-checkbox:checked+span, .label-radio .input-checkbox:checked+span
	{
	background-image: url(/images/icons/icon-checkbox-on.png);
	position: absolute;
	left: 0;
	top: 50%;
	margin-top: -12px;
	width: 24px;
	height: 24px;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-position-x: 50%;
	background-position-y: 50%;
	background-size: 24px;
}

.question-section em {
	font-size: 16px;
	font-weight: 700;
	color: #999;
}

.question-section h2 {
	position: relative;
	padding: 10px 0 28px;
	line-height: 1.2;
	font-size: 24px;
	font-weight: 700;
	color: #333;
	border-bottom: 1px solid #eee;
}

.survey-question-footer {
	overflow: hidden;
	margin: 20px auto 0;
	width: 450px;
}

.survey-question-footer>.btn {
	-webkit-appearance: none;
	appearance: none;
	display: inline-block;
	padding: 0 5px;
	min-width: 70px;
	height: 50px;
	line-height: 48px;
	font-size: 16px;
	font-weight: 700;
	text-align: center;
	outline: none;
	border: 1px solid #d7d7d7;
	background-color: #fff;
	vertical-align: middle;
	box-sizing: border-box;
}

.survey-question-footer>.btn:last-child {
	float: right;
	width: 200px;
}

.survey-question-footer>.btn:first-child {
	float: left;
	width: 200px;
}

.survey-question-footer>.btn.btn-dark, .btn.btn-colored {
	line-height: 50px;
}

.survey-question-footer>.btn-dark {
	background-color: #333;
	border: none;
	color: #fff;
}

.question-answer-wrap {
	padding-top: 15px;
	padding-bottom: 28px;
}

.question-answer-wrap>.label-question {
	padding-left: 44px;
	line-height: 20px;
}

.question-answer-wrap>.label-radio {
	position: relative;
	display: inline-block;
	padding-left: 36px;
	line-height: 24px;
	min-height: 24px;
	font-size: 16px;
	color: #333;
}

#question_ans {
	font-size: var(- -font-form-element-font-size, inherit) !important;
	line-height: var(- -font-form-element-line-height, 25px) !important;
	letter-spacing: var(- -font-form-element-letter-spacing, normal)
		!important;
	font-family: var(- -font-form-element-font-family, Circular,
		-apple-system, BlinkMacSystemFont, Roboto, Helvetica Neue,
		sans-serif) !important;
	text-transform: var(- -font-form-element-text-transform, undefined)
		!important;
	color: var(- -font-form-element-color, #484848) !important;
	padding-top: var(- -font-form-element-padding-top, undefined) !important;
	padding-bottom: var(- -font-form-element-padding-bottom, undefined)
		!important;
	background-color: var(- -color-input-background, #ffffff) !important;
	border-width: var(- -border-form-element-border-width, 1px) !important;
	border-color: var(- -color-input-border, #EBEBEB) !important;
	border-radius: var(- -border-form-element-border-radius, 4px) !important;
	margin-bottom: var(- -spacing-form-element-margin-bottom, 8px)
		!important;
	position: relative !important;
	border-style: solid !important;
	margin-right: auto !important;
	margin-left: auto !important;
	display: table !important;
	width: 45% !important;
	height: 40px;
}

#result-content {
	display: table;
	margin-right: auto;
	margin-left: auto;
	text-align: center;
	width: 70%;
	text-align: center;
	color: #333;
}

#result-content h2 {
	position: relative;
	padding: 10px 0 28px;
	font-size: 24px;
	color: #333;
	border-bottom: 1px solid #eee;
	line-height: 1;
	margin-top: 0px;
	margin-bottom: 0px;
	padding-top: 0px;
	padding-bottom: 10px;
}

svg {
	width: 350px;
	height: 350px;
}

#disimg {
	width: 100px;
	height: 100px;
}

#result-btn {
	display: table;
	margin-right: auto;
	margin-left: auto;
}

/* add default color for animation start  */

/* toggle this class */
.color-bg-start {
	background-color: salmon;
}

/* toggle class bg-animate-color */
.bg-animate-color {
	animation: random-bg .5s linear infinite;
}

/* add animation to bg color  */
@keyframes random-bg {from { filter:hue-rotate(0);
	
}

to {
	filter: hue-rotate(360deg);
}

}
.fun-btn {
	/* change bg color to get different hues    */
	background-color: #C3D69B;
	color: white;
	padding: 10 10;
	border: none;
	transition: all .3s ease;
	border-radius: 20px;
	line-height: 48px;
	text-transform: uppercase;
	outline: none;
	align-self: center;
	cursor: pointer;
	font-weight: bold;
	font-size: 20px;
	width: 200px;
}

.fun-btn:hover {
	animation: random-bg .5s linear infinite, grow 1300ms ease infinite;
}

.start-fun {
	background-color: #fff !important;
	/* change color of button text when fun is started   */
	color: salmon !important;
}

/* pulsating effect on button */
@keyframes grow {
  0% {
    transform: scale(1);
  }
  14% {
    transform: scale(1.3);
  }
  28% {
    transform: scale(1);
  }
  42% {
    transform: scale(1.3);
  }
  70% {
    transform: scale(1);
  }
}
</style>
</head>
<body>


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

	<%
		float[][] list = (float[][]) session.getAttribute("regiondata");
		String[] regionname_arr = {"강남", "강동", "강북", "관악", "광진", "구로", "금천", "노원", "동대문", "동자", "마포", "서대문", "성북",
				"송파", "양천", "영등포", "용산", "은평", "종로", "중", "중랑"};

		/* float[] region = list[(Integer.parseInt(request.getParameter("status_num"))-1)];
		String regionname= regionname_arr[(Integer.parseInt(request.getParameter("status_num"))-1)]; */
	%>
	<div class="survey-contents">
		<div class="survey-contents-inner">
			<%
				int question_ans[][] = null;
			%>
			<%
				if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("0")) {
			%>
			<div class="start-header">
				<h2>
					내일 미세먼지가 적은<span class="colored"> 지역구</span>를 추천해드립니다!
				</h2>
				<span><img src="images/basiclogo.png" class="Mainicon"></span>
			</div>
			<div class="start-intro">
				<p>
					내일미세만의 설문조사를 통해<br> 내일 미세먼지가 가장 적은 지역구를 알아보세요. <br> <br>
					설문은 약 1분정도 소요됩니다.
				</p>
				<input type="button" value="설문 시작하기"
					class="btn btn btn-dark btn-radius btn-lg"
					onclick="getSurvey( <%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)">
			</div>
			<%
				}
			%>
			<%
				if (request.getParameter("question_num") != null
						&& request.getParameter("question_num").equals("0") == false) {
			%>
			<nav class="survey-test-header">
				<ul class="step-list-wrap">
					<li>Step1<span class="glyphicon glyphicon-chevron-right"></span></li>
					<li>Step2<span class="glyphicon glyphicon-chevron-right"></span></li>
					<li>Step3</li>
				</ul>
			</nav>
			<%
				}
			%>
			<%
				if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("1")) {
			%>
			<div>
				<div>
					<div id="survey-question-contents">
						<section class="question-section">
							<em id="qNum">질문 <%=request.getParameter("question_num")%></em>
							<h2 id="qTitle">당신은 어느 지역구에 사시나요?</h2>
							<br>
							<div class="survey-content">
								<select id="question_ans" name="question_ans1" size=1 required>
									<option value=0 selected>지역구를 선택하세요</option>
									<option value="종로">종로구</option>
									<option value="중랑">중랑구</option>
									<option value="서대문">서대문구</option>
									<option value="영등포">영등포구</option>
									<option value="중">중구</option>
									<option value="성북">성북구</option>
									<option value="마포">마포구</option>
									<option value="동작">동작구</option>
									<option value="용산">용산구</option>
									<option value="강북">강북구</option>
									<option value="양천">양천구</option>
									<option value="관악">관악구</option>
									<option value="광진">광진구</option>
									<option value="노원">노원구</option>
									<option value="구로">구로구</option>
									<option value="강남">강남구</option>
									<option value="동대문">동대문구</option>
									<option value="은평">은평구</option>
									<option value="금천">금천구</option>
									<option value="송파">송파구</option>
									<option value="강동">강동구</option>
								</select><br>
							</div>
							<div class="survey-question-footer">
								<input type="button" value="이전" id="btn-prev"
									class="btn btn-radius"
									onclick="Survey_save(getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) - 1)%>)">
								<input type="button" value="다음" id="btn-next"
									class="btn btn-radius btn-dark"
									onclick="Survey_save('question_ans',<%=Integer.parseInt(request.getParameter("question_num")) - 1%>);getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)">
							</div>
						</section>
					</div>
					<%
						}
					%>

					<%
						if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("2")) {
					%>

					<div id="survey-question-contents">
						<section class="question-section">
							<em id="qNum">질문<%=request.getParameter("question_num")%>
							</em>
							<h2 id="qTitle">연령대는 어떻게 되십니까?</h2>
							<ul class="question-answer-wrap">

								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE=20> <span></span>
										20대
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE=30> <span></span>
										30대
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE=40> <span></span>
										40대
								</label></li>
							</ul>
							<div class="survey-question-footer">
								<input type="button" value="이전" id="btn-prev"
									class="btn btn-radius"
									onclick="getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) - 1)%>)">
								<input type="button" value="다음" id="btn-next"
									class="btn btn-radius btn-dark"
									onclick="Survey_save_radio('question_ans',<%=Integer.parseInt(request.getParameter("question_num")) - 1%>);getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)">

							</div>
						</section>
					</div>
					<%
						}
					%>

					<%
						if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("3")) {
					%>
					<div id="survey-question-contents">
						<section class="question-section">
							<em id="qNum">질문 <%=request.getParameter("question_num")%>
							</em>
							<h2 id="qTitle">나들이 유형은 어떻게 되시나요?</h2>
							<ul class="question-answer-wrap">
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE="혼자"> <span></span>
										혼자
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE="연인"> <span></span>
										연인
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE="친구"> <span></span>
										친구
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE="가족"> <span></span>
										가족
								</label></li>
								<li><label class="label-radio label-question"> <input
										TYPE=RADIO NAME=question_ans VALUE="모임"> <span></span>
										모임
								</label></li>
							</ul>
							<div class="survey-question-footer">
								<input type="button" value="이전" id="btn-prev"
									class="btn btn-radius"
									onclick="getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) - 1)%>)">
								<input type="button" value="다음" id="btn-next"
									class="btn btn-radius btn-dark"
									onclick="Survey_save_radio('question_ans',<%=Integer.parseInt(request.getParameter("question_num")) - 1%>);getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)">
							</div>
						</section>
					</div>
					<%
						}
					%>
					<%
						if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("4")) {
					%>
					<div id="survey-question-contents">
						<section class="question-section">
							<em id="qNum">질문 <%=request.getParameter("question_num")%>
							</em>
							<h2 id="qTitle">회원가입하시겠습니까?</h2>
							<div class="survey-question-footer">
								<input type="button" value="네, 좋아요! " id="btn-prev"
									class="btn btn-radius"
									onclick="window.location.href='/TeamProject/jsp/Join.jsp'">
								<input type="button" value="아니요, 다음에 할게요" id="btn-next"
									class="btn btn-radius btn-dark"
									onclick="Survey_save_radio('question_ans',<%=Integer.parseInt(request.getParameter("question_num")) - 1%>);getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)">
							</div>
						</section>
					</div>
					<%
						}
					%>

					<%
						if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("5")) {
					%>
					<%
						if (session.getAttribute("login-status") != null) {
					%>
					<script>window.alert(<%=(String) session.getAttribute("login-status")%>)</script>
					<%
						} else {
					%>
					<%
						}
					%>
					<div class="start-header">
						<h2>
							설문조사가 <span class="colored">완료</span>되었습니다!
						</h2>
						<span><img src="images/basiclogo.png" class="Mainicon"></span>
					</div>
					<div class="start-intro">
						<br> <br> <br> <input type="button"
							value="결과 확인하러 가기" class="btn btn btn-dark btn-radius btn-lg"
							onclick="Show();getResult(6,question_ans[0])">
					</div>
					<%
						}
					%>

					<%-- 설문조사 결과! --%>

					<%
						if (request.getParameter("question_num") != null && request.getParameter("question_num").equals("6")) {
							String region = request.getParameter("region");
							int idx = 0;
							for (int i = 0; i < regionname_arr.length; i++) {
								if (regionname_arr[i].equals(request.getParameter("region"))) {
									idx = i;
									break;
								}
							}
							float regionDust = list[idx][hour];
							String dustStatus = "";
							if (regionDust > 100)
								dustStatus = "매우나쁨";
							else if (regionDust > 50)
								dustStatus = "나쁨";
							else if (regionDust > 30)
								dustStatus = "보통";
							else
								dustStatus = "좋음";
					%>
					<div id="result-content">
						<div id="result-district">
							<h1>
								<strong><%=region%>구</strong>
							</h1>
							<!-- ------------------------------------------------------------------------------ -->
							<%
								if (regionDust > 100) {
							%>
							<svg xmlns="http://www.w3.org/2000/svg"
								xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
								y="0px" viewBox="0 0 100 100"
								enable-background="new 0 0 100 100" xml:space="preserve">
								<g>
								<path fill="#f65e63"
									d="M49.283,10.853c-21.586,0-39.147,17.562-39.147,39.148s17.561,39.147,39.147,39.147   c21.586,0,39.147-17.562,39.147-39.148S70.869,10.853,49.283,10.853z M49.283,84.148C30.454,84.148,15.136,68.83,15.136,50   c0-18.829,15.318-34.148,34.147-34.148S83.431,31.171,83.431,50C83.431,68.83,68.112,84.148,49.283,84.148z" /></g>
								<g>
								<path fill="#f65e63"
									d="M42.202,36.144c-0.878-0.879-2.304-0.879-3.182,0l-3.43,3.43l-3.43-3.43c-0.878-0.879-2.304-0.879-3.182,0   c-0.879,0.878-0.879,2.303,0,3.182l3.43,3.43l-3.43,3.43c-0.879,0.878-0.879,2.303,0,3.182c0.439,0.439,1.015,0.659,1.591,0.659   s1.152-0.22,1.591-0.659l3.43-3.43l3.43,3.43c0.439,0.439,1.015,0.659,1.591,0.659s1.152-0.22,1.591-0.659   c0.879-0.878,0.879-2.303,0-3.182l-3.43-3.43l3.43-3.43C43.081,38.447,43.081,37.022,42.202,36.144z" />
								<path fill="#f65e63"
									d="M65.711,42.755l3.43-3.43c0.878-0.878,0.878-2.303-0.001-3.182c-0.879-0.879-2.304-0.879-3.183,0l-3.429,3.43l-3.429-3.43   c-0.879-0.879-2.304-0.879-3.183,0c-0.879,0.878-0.879,2.303-0.001,3.182l3.43,3.43l-3.43,3.43   c-0.878,0.878-0.878,2.303,0.001,3.182c0.439,0.439,1.015,0.659,1.591,0.659s1.152-0.22,1.592-0.659l3.429-3.43l3.429,3.43   c0.439,0.439,1.016,0.659,1.592,0.659s1.151-0.22,1.591-0.659c0.879-0.878,0.879-2.303,0.001-3.182L65.711,42.755z" /></g>
								<path fill="#f65e63"
									d="M66.854,57.732c-3.637,0-6.596,2.944-6.627,6.573c0,0.01-0.003,0.019-0.003,0.028c0,1.175-0.956,2.13-2.13,2.13  c-1.176,0-2.131-0.955-2.131-2.13c0-0.01-0.003-0.019-0.003-0.028c-0.031-3.629-2.991-6.573-6.627-6.573  c-3.637,0-6.596,2.944-6.627,6.573c0,0.01-0.003,0.019-0.003,0.028c0,1.175-0.956,2.13-2.13,2.13s-2.13-0.955-2.13-2.13  c0-0.01-0.003-0.019-0.003-0.028c-0.031-3.629-2.991-6.573-6.627-6.573c-3.656,0-6.63,2.975-6.63,6.63c0,1.242,1.007,2.25,2.25,2.25  s2.25-1.008,2.25-2.25c0-1.175,0.956-2.13,2.13-2.13s2.13,0.955,2.13,2.13c0,0.01,0.003,0.019,0.003,0.028  c0.031,3.629,2.991,6.573,6.627,6.573s6.597-2.944,6.627-6.573c0-0.01,0.003-0.019,0.003-0.028c0-1.175,0.956-2.13,2.13-2.13  c1.175,0,2.13,0.955,2.13,2.13c0,0.01,0.003,0.019,0.003,0.028c0.031,3.629,2.991,6.573,6.628,6.573s6.596-2.944,6.627-6.573  c0-0.01,0.003-0.019,0.003-0.028c0-1.175,0.956-2.13,2.13-2.13c1.176,0,2.131,0.955,2.131,2.13c0,1.242,1.008,2.25,2.25,2.25  s2.25-1.008,2.25-2.25C73.484,60.707,70.51,57.732,66.854,57.732z" />
								</svg>
							<%
								} else if (regionDust > 50) {
							%>
							<svg xmlns="http://www.w3.org/2000/svg"
								xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
								y="0px" viewBox="0 0 100 100"
								enable-background="new 0 0 100 100" xml:space="preserve">
								<g>
								<path fill="#f8d75d"
									d="M49.95,10.853c-21.586,0-39.147,17.562-39.147,39.148S28.364,89.148,49.95,89.148S89.098,71.587,89.098,50   S71.536,10.853,49.95,10.853z M49.95,84.148C31.121,84.148,15.803,68.83,15.803,50c0-18.829,15.318-34.148,34.147-34.148   S84.098,31.171,84.098,50C84.098,68.83,68.779,84.148,49.95,84.148z" />
								<path fill="#f8d75d"
									d="M35.143,46.544c2.865,0,5.196-2.331,5.196-5.196c0-2.865-2.331-5.196-5.196-5.196s-5.196,2.331-5.196,5.196   C29.946,44.213,32.277,46.544,35.143,46.544z" />
								<path fill="#f8d75d"
									d="M64.758,46.544c2.865,0,5.196-2.331,5.196-5.196s-2.331-5.196-5.196-5.196s-5.196,2.331-5.196,5.196   S61.893,46.544,64.758,46.544z" />
								<path fill="#f8d75d"
									d="M49.95,56.506c-9.442,0-17.855,4.419-20.235,8.514c-0.694,1.193-0.288,2.724,0.905,3.417   c0.396,0.23,0.828,0.339,1.254,0.339c0.862,0,1.7-0.445,2.164-1.244c1.257-2.162,7.649-6.025,15.912-6.025   c8.109,0,14.185,3.875,16.11,6.317c0.855,1.084,2.425,1.271,3.512,0.416c1.084-0.855,1.271-2.427,0.416-3.512   C66.865,60.768,59.264,56.506,49.95,56.506z" /></g>
								</svg>
							<%
								} else if (regionDust > 30) {
							%>
							<svg xmlns="http://www.w3.org/2000/svg"
								xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
								y="0px" viewBox="0 0 100 100"
								enable-background="new 0 0 100 100" xml:space="preserve">
								<g>
								<path fill="#bde77a"
									d="M50.95,10.853c-21.586,0-39.147,17.562-39.147,39.148S29.364,89.148,50.95,89.148S90.098,71.587,90.098,50   S72.536,10.853,50.95,10.853z M50.95,84.148C32.121,84.148,16.803,68.83,16.803,50c0-18.829,15.318-34.148,34.147-34.148   S85.098,31.171,85.098,50C85.098,68.83,69.779,84.148,50.95,84.148z" />
								<path fill="#bde77a"
									d="M36.143,46.544c2.865,0,5.196-2.331,5.196-5.196c0-2.865-2.331-5.196-5.196-5.196s-5.196,2.331-5.196,5.196   C30.946,44.213,33.277,46.544,36.143,46.544z" />
								<path fill="#bde77a"
									d="M65.758,46.544c2.865,0,5.196-2.331,5.196-5.196s-2.331-5.196-5.196-5.196s-5.196,2.331-5.196,5.196   S62.893,46.544,65.758,46.544z" />
								<path fill="#bde77a"
									d="M69.023,52.34c-1.381,0-2.5,1.119-2.5,2.5c0,8.309-6.695,14.574-15.573,14.574c-8.878,0-15.574-6.266-15.574-14.574   c0-1.381-1.119-2.5-2.5-2.5s-2.5,1.119-2.5,2.5c0,11.159,8.845,19.574,20.574,19.574c11.729,0,20.573-8.415,20.573-19.574   C71.523,53.459,70.404,52.34,69.023,52.34z" /></g>
								</svg>
							<%
								} else {
							%>
							<svg xmlns="http://www.w3.org/2000/svg"
								xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
								y="0px" viewBox="0 0 100 100"
								enable-background="new 0 0 100 100" xml:space="preserve">
									<g>
									<path fill="#62bcef"
									d="M49.467,10.853c-21.586,0-39.147,17.562-39.147,39.148s17.562,39.148,39.147,39.148S88.614,71.587,88.614,50   S71.053,10.853,49.467,10.853z M49.467,84.148C30.638,84.148,15.319,68.83,15.319,50c0-18.829,15.318-34.148,34.147-34.148   S83.614,31.171,83.614,50C83.614,68.83,68.296,84.148,49.467,84.148z" />
									<path fill="#62bcef"
									d="M35.268,49.917c3.345,0,7.616-5.594,7.616-9.976c0-3.042-1.687-5.166-4.103-5.166c-1.564,0-2.79,0.785-3.477,2.165   c-0.678-1.436-1.92-2.254-3.52-2.254c-2.415,0-4.102,2.124-4.102,5.165C27.683,44.272,31.936,49.917,35.268,49.917z" />
									<path fill="#62bcef"
									d="M67.147,34.776c-1.564,0-2.79,0.785-3.477,2.165c-0.678-1.436-1.92-2.254-3.52-2.254c-2.415,0-4.102,2.124-4.102,5.165   c0,4.421,4.253,10.066,7.585,10.066c3.346,0,7.616-5.594,7.616-9.976C71.251,36.9,69.563,34.776,67.147,34.776z" />
									<path fill="#62bcef"
									d="M67.54,53.817c-1.381,0-2.5,1.119-2.5,2.5c0,8.309-6.695,14.574-15.573,14.574c-8.878,0-15.574-6.266-15.574-14.574   c0-1.381-1.119-2.5-2.5-2.5s-2.5,1.119-2.5,2.5c0,11.159,8.845,19.574,20.574,19.574c11.728,0,20.573-8.415,20.573-19.574   C70.04,54.937,68.921,53.817,67.54,53.817z" /></g></svg>
							<%
								}
							%>

							<!-- -------------------------------------------------------  -->
						</div>
						<div id="result-script">
							<h2>
								선택하신 지역구의 내일 미세먼지는 <strong><%=dustStatus%></strong>입니다.
							</h2>
							<h2>
								미세먼지 수치는 <strong><%=regionDust%></strong>입니다.
							</h2>

						</div>
						<%
							if (regionDust < 50) {
						%>
						<div class="survey-question-footer">
							<input type="button" onclick="MoveUrl();" value="인기맛집 보러가기"
								id="btn-next" class="btn btn-radius btn-dark">
						</div>
						<%
							} else {
						%>
						<div class="survey-question-footer">
							<input type="button" onclick="MoveUrl();" value="그래도 인기맛집 보러가기"
								id="btn-prev" class="btn btn-radius"> <input
								type="button" id="btn-next" class="btn btn-radius btn-dark"
								onclick="getSurvey(<%=String.valueOf(Integer.parseInt(request.getParameter("question_num")) + 1)%>)"
								value="미세먼지 낮은 지역구 조회">
						</div>
						<%
							}
						%>
					</div>
					<%
						}
					%>

					<!-- 결과에대한 선택 3번  -->
					<%
						if (session.getAttribute("survey-result") != null && request.getParameter("question_num") != null
								&& request.getParameter("question_num").equals("7")) {
							CustomInfoVO vo = (CustomInfoVO) session.getAttribute("survey-result");
							float[] timeDustData = new float[list.length];
							int regionidx = 0;
							for (int i = 0; i < list.length; i++) {
								timeDustData[i] = list[i][hour];
							}
							float[] sorttimeDD = new float[list.length];
							for (int i = 0; i < list.length; i++) {
								sorttimeDD[i] = list[i][hour];
							}
							Arrays.sort(sorttimeDD);
							String[] sortregionname = new String[3];
							for (int j = 0; j < 3; j++) {
								for (int k = 0; k < timeDustData.length; k++) {
									if (timeDustData[k] == sorttimeDD[j]) {
										sortregionname[j] = regionname_arr[k];
									}
								}
							}
					%>
					<div id="result-content">
						<h2 style="line-height: 1.2;">
							<strong>미세먼지가 적은 지역구 순위</strong><br>
							바로 인기맛집 보러가요!
						</h2>
						<script type="text/javascript">
		$('.fun-btn').on(
				'click',
				function(event) {
					$(this).toggleClass('start-fun');
					var $page = $('.page');
					$page.toggleClass('color-bg-start').toggleClass(
							'bg-animate-color');

					//change text when when button is clicked

					$(this).hasClass('start-fun') ? $(this)
							.text('stop the fun') : $(this).text(
							'start the fun');

				});
	</script>
						<div class="page">
						<br>
							<input class="fun-btn" type="button" name=question_ans
								value="1위 <%=sortregionname[0]%>구"
								onclick="regionUpdate('<%=sortregionname[0]%>');MoveUrl()"><br><br>
							<input class="fun-btn" type="button" name=question_ans
								value="2위  <%=sortregionname[1]%>구"
								onclick="regionUpdate('<%=sortregionname[1]%>');MoveUrl()"><br><br>
							<input class="fun-btn" type="button" name=question_ans
								value="3위  <%=sortregionname[2]%>구"
								onclick="regionUpdate('<%=sortregionname[2]%>');MoveUrl()">
						</div>
					</div>
					<%
						}
					%>

				</div>
			</div>
		</div>
	</div>
</body>

