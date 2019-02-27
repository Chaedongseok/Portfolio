<%@page import="Model.vo.SearchPlaceVO"%>
<%@page import="Model.vo.CustomInfoVO"%>
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
<title>내일미세</title>
<style type="text/css">
body {
	background-color: #fafafa;
}

.bottom-footer {
	height: auto;
	min-height: 100%;
}

.myplace-view {
	padding-top: 60px;
}

#myplace {
	margin: 0 auto;
	width: 908px;
}

.myplace-header {
	margin-bottom: 15px;
}

.myplace-header h2 {
	float: left;
	margin-top: 2px;
	font-size: 28px;
	font-weight: 400;
	color: #333;
}

.clearfix {
	zoom: 1;
}

.btn-wrap {
	float: right;
}

.btn {
	-webkit-appearance: none;
	appearance: none;
	display: inline-block;
	padding: 0 36px;
	min-width: 74px;
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

button {
	cursor: pointer;
}

.btn-add-myplace {
	padding: 0 30px;
	color: rgba(153, 153, 153, 0.7);
	font-weight: 700;
}

.btn-remove-myplace {
	margin-left: 20px;
	padding: 0 20px;
	color: rgba(153, 153, 153, 0.7);
	font-weight: 700;
}

.btn-small {
	height: 40px;
	line-height: 38px;
	font-size: 14px;
	border-radius: 20px;
}

.btn-radius {
	border-radius: 25px;
}

ul, li {
	list-style: none;
}

.bucket-list-wrap li:first-child {
	border-top: none;
}

.myplace-list-wrap li {
	padding: 20px 0;
	border-top: 1px solid #eee;
	display: table;
	width: 100%;
}

p {
	display: inline-block;
	line-height: 1;
	font-size: 18px;
	font-weight: 400;
	border-bottom: solid 1px #848484;
	padding-bottom: 1px;
	margin-right: 20px;
}

a {
	text-decoration: none;
	color: inherit;
}

#myplacemap {
	margin-top: 200px;
	margin-right: auto;
	margin-left: auto;
	width: 500px;
	height: 400px;
}
</style>
</head>
<body>
	<div id="app" class="bottom-footer">
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
						<li><a href="/TeamProject/jsp/SearchPlace.jsp" id ="menu-name2">플레이스 탐색</a></li>
						
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
		<div id="myplacemap" style="width: 900px; height: 400px;"></div>
		<script
			src="//dapi.kakao.com/v2/maps/sdk.js?appkey=8f02b495fe5dddc08ecbbe44306cb90a"></script>
		<script>
			var mapContainer = document.getElementById('myplacemap'), // 지도를 표시할 div 
			mapOption = {
				center : new daum.maps.LatLng(37.565991, 126.976863), // 지도의 중심좌표
				level : 9, // 지도의 확대 레벨
				mapTypeId : daum.maps.MapTypeId.ROADMAP
			// 지도종류
			};

			// 지도를 생성한다 
			var map = new daum.maps.Map(mapContainer, mapOption);

			// 지도에 확대 축소 컨트롤을 생성한다
			var zoomControl = new daum.maps.ZoomControl();

			// 지도의 우측에 확대 축소 컨트롤을 추가한다
			map.addControl(zoomControl, daum.maps.ControlPosition.RIGHT);
			
			var imageSrc = "http://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png"; 
			
			var imageSize = new daum.maps.Size(24, 35); 
			var markerImage = new daum.maps.MarkerImage(imageSrc, imageSize); 
			 
			 
			
			var markers = [];
			function addMarker(position) {
			    
			    // 마커를 생성합니다
			    var marker = new daum.maps.Marker({
			        position: position,
			        image : markerImage
			    });

			    // 마커가 지도 위에 표시되도록 설정합니다
			    marker.setMap(map);
			    
			    // 생성된 마커를 배열에 추가합니다
			    markers.push(marker);
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
</body>
</html>
<main id="main" class="myplace-view" role="main">

<div id="myplace">
	<div class="myplace-header clearfix">
		<h2>마이 플레이스</h2>
		<div class="btn-wrap">
			<a href="/TeamProject/jsp/SearchPlace.jsp"
				class="btn btn-radius btn-small btn-icon btn-add-place"><span>장소
					찜하러하기</span></a>
			<button class="btn btn-radius btn-small btn-icon btn-remove-place">
				<span>비우기</span>
			</button>
		</div>
	</div>
	<%if(request.getAttribute("list")!=null){
	ArrayList<SearchPlaceVO> list= (ArrayList<SearchPlaceVO>)request.getAttribute("list");

	
	%>
	
		<ul class="myplace-list-wrap">
		<%for(SearchPlaceVO cvo : list){ %>
			<li class="" data="PDPMO159KRV" data-limitation="2"
				data-nick="milkthistle">
				<div class="myplace-list-row">
					<p>
						<a href="#" onclick="window.location.href='http://70.12.113.169:8000/TeamProject/Recommend?action=showdetail&content-code=<%=cvo.getContent_code()%>'"><%=cvo.getContent_name()%></a>
					</p>
					<span><%=cvo.getNewaddress() %></span>
					<script>addMarker(new daum.maps.LatLng(<%=cvo.getY()%>,<%=cvo.getX()%>))</script>
				</div>
			</li>
		<%} %>
		</ul>
	<%} %>
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