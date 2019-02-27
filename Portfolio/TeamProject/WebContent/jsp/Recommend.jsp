<%@page import="Model.vo.RecommendVO"%>
<%@page import="java.util.ArrayList"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="stylesheet"
	href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">
<link rel="stylesheet"
	href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap-theme.min.css">
<link rel="stylesheet" href="/TeamProject/jsp/css/DustTotal.css">
<link rel="stylesheet" href="/TeamProject/jsp/css/FooterTotal.css">
<script src="//code.jquery.com/jquery-1.11.0.min.js"></script>
<script type="text/javascript"
	src="//dapi.kakao.com/v2/maps/sdk.js?appkey=8f02b495fe5dddc08ecbbe44306cb90a"></script>
<title>내일 미세</title>
<style type="text/css">
element.style {
	position: fixed;
}

#app.survey-result {
	background-color: #fafafa;
}

.survey-result-view {
	overflow: hidden;
	margin: 0 auto;
	padding-top: 150px;
	padding-bottom: 30px;
	width: 1140px;
}

.left-menu-layout {
	overflow: hidden;
	margin: 0 auto;
	width: 1140px;
}

.left-menu {
	float: left;
	width: 192px;
}

.left-menu li {
	padding: 8px;
}

.left-menu li>span {
	padding-left: 20px;
	font-size: 18px;
	vertical-align: top;
	text-align: left;
}

.left-menu-layout .left-menu li {
	text-align: center;
	padding: 10px 0;
}

.left-menu ul {
	padding: 10px;
}

.left-menu.survey-result h3 {
	font-weight: 700;
	color: #333;
	text-align: center;
	border-top: 1px solid #adadad;
	border-bottom: 1px solid #adadad;
}

.colored {
	color: #f1574b;
}

.right-contents {
	position: relative;
	float: right;
	margin-left: 300px;
	width: 918px;
}

ul {
	list-style: none;
}

.menu-btns {
	text-align: center;
}

.contents-header {
	position: relative;
	padding: 35px 30px 45px;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-size: cover;
}

.contents-main {
	position: relative;
	padding: 30px;
	width: 100%;
	box-sizing: border-box;
}

.contents-item.first {
	border-bottom: 2px solid #eee;
}

.item-title {
	margin-bottom: 45px;
	font-size: 18px;
	color: #333;
}

survey-result li {
	position: relative;
	display: table;
	width: 100%;
	height: 30px;
}

.survey-result li>p {
	display: table-cell;
	padding-left: 20px;
	font-size: 18px;
	vertical-align: top;
	text-align: left;
}

table {
	border: 1px solid black;
}

ul {
	display: block;
	list-style-type: disc;
	margin-block-start: 1em;
	margin-block-end: 1em;
	margin-inline-start: 0px;
	margin-inline-end: 0px;
	padding-inline-start: 0px;
}

#map {
	position: relative;
	float: right;
	margin-left: 300px;
	width: 918px;
}

#listicon {
	width: 25px;
	background-repeat: no-repeat;
	padding-top: 0px;
}
</style>
</head>
<body>
	<script type="text/javascript" charset="utf-8">
		function getContent(Menu_num) {/* Survey_content.jsp에서 내용물을 끄집어와 Ajax로 보여줌 */
			if (window.XMLHttpRequest) {
				var xmlhttp = new XMLHttpRequest();
			} else { // code for IE6, IE5
				var xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
			}
			xmlhttp.open("GET", "/TeamProject/jsp/RecommendContent.jsp?Menu_num="
					+ Menu_num, true);
			xmlhttp.onreadystatechange = function() {
				if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
					document.getElementById("content-main").innerHTML = xmlhttp.responseText;
				}
			}
			xmlhttp.send();
		}
		getContent(0);
		function Mapadapt(y,x){
			var mapContainer = document.getElementById('map'), // 지도를 표시할 div 
			mapOption = {
				center : new daum.maps.LatLng(37.565991, 126.976863), // 지도의 중심좌표
				level : 9
			// 지도의 확대 레벨
			};

			var map = new daum.maps.Map(mapContainer, mapOption); // 지도를 생성합니다

			// 마커가 표시될 위치입니다 
			var markerPosition = new daum.maps.LatLng(y, x);

			// 마커를 생성합니다
			var marker = new daum.maps.Marker({
				position : markerPosition
			});

			// 마커가 지도 위에 표시되도록 설정합니다
			marker.setMap(map);

			// 아래 코드는 지도 위의 마커를 제거하는 코드입니다
			// marker.setMap(null);
		}
		var prevScrollpos = window.pageYOffset;
		
		window.onload=function(){
			window.onscroll = function() {
			var currentScrollPos = window.pageYOffset;
			  if (prevScrollpos > currentScrollPos) {
			    document.getElementById("navbar").style.background = "rgba(0, 255, 0, 0)";
				
			    document.getElementById("menu-name2").style.color = "white";
			    
			    
			    if(document.getElementById("menu-name4")!=null){
			    	document.getElementById("menu-name3").style.color = "white";
			   		document.getElementById("menu-name4").style.color = "white";
			    }
			    if(document.getElementById("menu-name5")!=null){
			    	document.getElementById("menu-name6").style.color = "white";
			    	document.getElementById("menu-name5").style.color = "white";
			    }
			
			  } else {
			    document.getElementById("navbar").style.background = "white";
			    
			    document.getElementById("menu-name2").style.color = "black";
			    if(document.getElementById("menu-name4")!=null){
			    	document.getElementById("menu-name3").style.color = "black";
			   		document.getElementById("menu-name4").style.color = "black";
			    }
			    if(document.getElementById("menu-name5")!=null){
			    	document.getElementById("menu-name6").style.color = "black";
			    	document.getElementById("menu-name5").style.color = "black";
			    }
			  }
			  prevScrollpos = currentScrollPos;
			}
		}
		
	</script>

	<%
		ArrayList<RecommendVO> list = (ArrayList<RecommendVO>) session.getAttribute("selectfive");
	%>

	<%--<% RecommendVO voinit = list.get(1); %>
<script>Mapadapt(<%=voinit.getY()%>,<%=voinit.getX()%>);</script> --%>



	<div id="app" class="bottom-footer survey-result">
		<header id="header">
		<div class="navbar-fixed-top navbar-default" id="navbar">
			<!-- navigation bar -->
			<div class="container-fluid">
				<!-- navigation에서 일반적인부분 -->
				<div class="navigation-header">
					<button type="button" class="navbar-toggle collapsed"
						data-toggle="collapse" data-target="#bs-example-navbar-collapse-1"
						aria-expanded="false" style="margin-top: 15px;">
						<span class="sr-only"></span> <span class="icon-bar"></span> <span
							class="icon-bar"></span> <span class="icon-bar"></span> <span
							class="icon-bar"></span> <span class="icon-bar"></span>
					</button>

					<a class="navbar-brand col-sm-1.5" href="/TeamProject/jsp/Main.jsp">
						<img src="images/textlogo1.png" class="Mainicon">
					</a>
				</div>
				<div class="collapse navbar-collapse"
					id="bs-example-navbar-collapse-1">
					<ul class="nav navbar-nav">
	
						<li><a href="SearchPlace.jsp" id ="menu-name2">플레이스 탐색</a></li>
						
						<%
							if (session.getAttribute("login-status") != null
									&& ((String) session.getAttribute("login-status")).equals("login")) {
						%>
						<li><a href="/TeamProject/CustomInfo?action=read" id ="menu-name3">마이 플레이스</a></li>
						<li><a href="/TeamProject/CustomInfo?action=logout" id ="menu-name4">로그아웃</a></li>
						<%
							} else {
						%>
						<li><a href="/TeamProject/jsp/Login.jsp" id ="menu-name6">마이 플레이스</a></li>
						<li><a href="/TeamProject/jsp/Login.jsp" id ="menu-name5">로그인</a></li>
						<%
							}
						%>
					</ul>
				</div>
			</div>
		</div>
	</header>

		<main id="main" class="survey-result-view" role="main">
		<div class="left-menulayout clearfix">
			<nav id="nav-floating"
				class="left-menu survey-result fixed-left-menu"
				style="position: fixed;">
				<h3 style="padding: 20px; font-size: 20px;">
					<div>
						인기 맛집 <span class="bold colored">TOP 5</span>
					</div>
				</h3>
				<ul>
					<%
						int cnt = 0;
					%>
					<%
						for (RecommendVO vo : list) {
					%>
					<a href="#"
						onclick="Mapadapt(<%=vo.getY()%>,<%=vo.getX()%>);getContent(<%=cnt%>);">
						<li><img id="listicon" src="images/spoon.png"> <span><%=vo.getPlace()%></span></li>

					</a>
					<%
						cnt++;
						}
					%>
				</ul>
				<div class="menu-btns">
					<div class="btn-right">
						<button class="btn btn-success btn-radius btn-lg btn-searchplace" onclick="window.location.href='/TeamProject/jsp/SearchPlace.jsp'">플레이스
							탐색</button>
					</div>
				</div>
			</nav>
			<div class="right-contents survey-result-contents" id=content-main></div>
			<div id="map" style="width: 900px; height: 400px;"></div>
		</div>
		</main>
	</div>
	<footer id="footer">
		<div>
			<div class="clearfix">
				<dl class="company-info info1">
					<dt>소속</dt>
					<dd>(주)멀티캠퍼스</dd>
					<dt>본사</dt>
					<dd>서울특별시 강남구 역삼동 테헤란로 212, 13층</dd>
					<dt>팀명</dt>
					<dd>내일미세</dd>
					<dt>팀원</dt>
					<dd>채동석, 신동의, 이혜란</dd>
				</dl>
				<dl class="company-info info2">
					<dt>고객문의</dt>
					<dd>
						<a href="mailto:cs@misetomorrow.us">cs@misetomorrow.us</a>
					</dd>
					<dt>제휴문의</dt>
					<dd>
						<a href="mailto:marketing@misetomorrow.us">marketing@misetomorrow.us</a>
					</dd>
					<dd>
						<img src="images/basiclogo.png" id="footer_logo" width="50"
							height="50">
					</dd>
					<dd>
						<p class="copyright">
							Copyright<span class="700">ⓒ 2018 Misetomorrow Co.</span>, Ltd.
							All rights reserved.
						</p>
					</dd>
				</dl>
			</div>
		</div>
	</footer>
</body>
</html>