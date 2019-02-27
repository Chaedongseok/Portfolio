<%@page import="java.util.Random"%>
<%@page import="java.util.HashMap"%>
<%@page import="Model.vo.SearchPlaceVO"%>
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
<title>내일미세</title>
<style type="text/css">
body {
	background-color: #fafafa;
}

em {
	font-style: normal;
}

section {
	display: block;
}

#header {
	position: relative;
	display: block;
}

#header.product-detail-page {
	height: 920px;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-size: cover;
}

.header-contents {
	padding-top: 150px;
	padding-left: 70px;
	padding-left: 70px;
	padding-bottom: 50px;
	height: 100%;
	text-align: left;
	font-size: 18px;
	box-sizing: border-box;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-size: cover;
	margin-right: 0px;
}

.header-contents.vitaminb {
	background-color: #8cc782;
}

.contentA {
	width: 300px;
	margin-right: 0px;
}

.contentA>h1 {
	margin-top: 17px;
	font-size: 36px;
	font-weight: 700;
	margin-right: 0px;
}

.contentA>h1>span {
	font-weight: 300;
	margin-right: 0px;
}

.contentA p.description {
	margin-top: 65px;
	width: 35%;
	line-height: 18px;
	font-size: 16px;
	font-weight: 400;
	word-break: keep-all;
	margin-right: 0px;
}

.contentA>.btn-wrap {
	margin-top: 53px;
	margin-right: 0px;
}

.contentA>.btn-wrap button {
	width: 300px;
	margin-right: 15px;
}

#item-listA li {
	float: left;
	width: auto;
	color: #333;
}

#item-listA li:first-child {
	margin-left: 0;
}

.btn.btn-dark, .btn.btn-colored {
	line-height: 50px;
}

.btn-colored {
	background-color: #f1574b;
	border: none;
	color: #000000;
}

.btn-shadow {
	box-shadow: 0 5px 4px 0 rgba(0, 0, 0, 0.15);
}

.btn-radius {
	border-radius: 25px;
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

.wrapper {
	margin: 0 auto;
	width: 1140px;
}

.product-effects-wrap {
	text-align: center;
}

.product-effects-wrap>em>span {
	display: inline-block;
	margin-top: 20px;
	font-size: 32px;
	font-weight: 300;
}

.product-effects-wrap>em {
	font-size: 52px;
	font-weight: 700;
	color: #333;
}

.product-effects-list-wrap {
	margin: 100px 0 200px;
}

.clearfix {
	zoom: 1;
}

ul, ol, li {
	list-style: none;
}

ul, menu, dir {
	display: block;
	list-style-type: disc;
	margin-block-start: 1em;
	margin-block-end: 1em;
	margin-inline-start: 0px;
	margin-inline-end: 0px;
	padding-inline-start: 0px;
}

.product-effects-list-wrap li {
	float: left;
	width: 340px;
	color: #333;
}

.product-effects-list-wrap li:first-child {
	margin-left: 0;
}

.type3 li {
	margin-left: 60px;
}

.type3 li>p {
	margin: 50px auto 0;
}

img {
	max-width: 100%;
	height: auto;
}

.product-effects-list-wrap li>em {
	display: block;
	margin-top: 40px;
	font-size: 29px;
	font-weight: 400;
	text-align: center;
}

.product-effects-list-wrap li>p {
	margin-top: 50px;
	line-height: 30px;
	font-size: 18px;
}

.product-box-image {
	overflow: hidden;
}

.product-box-image img {
	float: left;
	width: 50%;
	vertical-align: top;
}

.product-intro-wrap {
	margin-top: 124px;
}

.product-intro-header {
	padding-top: 80px;
	padding-bottom: 100px;
	text-align: center;
	background-size: 84px;
	background-repeat: no-repeat;
	background-position: 50% 0;
}

.product-intro-header em {
	display: block;
	font-size: 32px;
	font-weight: 300;
	color: #333;
}

.product-intro-header span {
	font-size: 52px;
	font-weight: 700;
}

.product-intro-inner {
	margin-bottom: 180px;
}

.product-intro-main img {
	vertical-align: top;
}

.product-detail {
	position: relative;
	margin-top: 60px;
	padding-left: 82px;
	min-height: 100px;
	color: #333;
	border-left: 2px solid #333;
}

.product-order {
	position: absolute;
	left: 38px;
	top: 0;
	font-size: 24px;
	font-weight: 700;
}

.product-detail em {
	line-height: 1;
	font-size: 35px;
	font-weight: 300;
	word-break: keep-all;
}

.medium {
	font-weight: 700 !important;
}

.product-detail p {
	margin-top: 40px;
	line-height: 32px;
	font-size: 18px;
}

.customer-review>h3 {
	margin-top: 50px;
	font-size: 1.17em;
	font-weight: 700;
}

.customer-review li {
	float: left;
	margin-top: 50px;
	width: 360px;
	overflow: hidden;
	border-radius: 6px 6px 0 0;
	border: 1px solid #e6e6e6;
	box-shadow: 0px 2px 3px 0 rgba(0, 0, 0, 0.08);
	background-color: #fff;
	margin-right: 16px;
}

.customer-review li .review-header {
	padding: 24px 20px 24px 20px;
}

.review-header>em {
	position: relative;
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
	display: inline-block;
	width: 100%;
	font-size: 14px;
	font-weight: 700;
}

.normal {
	font-weight: 400 !important;
}

.review-header>span {
	position: relative;
	display: block;
	float: right;
	margin-top: 8px;
	font-size: 12px;
}

.review-image {
	position: relative;
	margin-top: 14px;
	padding-bottom: 100%;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-size: cover;
}

.customer-review li.has-image .open-contents {
	height: 95px;
}

.customer-review li .open-contents {
	padding: 0 12px;
	height: 469px;
	max-height: 469px;
	overflow: hidden;
	text-overflow: ellipsis;
	display: block;
	display: -webkit-box;
	-webkit-line-clamp: 4;
	-webkit-box-orient: vertical;
}

.customer-review li.has-image .open-contents>p {
	margin-top: 20px;
}

.customer-review li .open-contents>p {
	margin-top: 10px;
	line-height: 19px;
	padding: 0 10px;
	font-size: 14px;
	font-weight: 300;
	color: #333;
	word-break: keep-all;
}

.customer-review li .open-more {
	margin: 15px 0;
	padding: 4px 22px 10px 10px;
	text-align: right;
	color: #f1574b;
	font-size: 12px;
	text-decoration: underline;
	cursor: pointer;
}
</style>
</head>
<body>
<script>
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
		SearchPlaceVO vo = (SearchPlaceVO) request.getAttribute("selectone");
	%>
	<%
		if (vo != null) {
	%>

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

<%
HashMap<Integer,String> colormap= new HashMap<Integer,String>();
colormap.put(1,"#8cc782");
colormap.put(2,"#a7a8d3");
colormap.put(3,"#ead499");
colormap.put(4,"#d3d1d1");
colormap.put(5,"#81b0ce");
colormap.put(6,"#e6aa7c");

Random rd = new Random();
int rnum = rd.nextInt(6);
rnum = rnum+1;
String backcol = colormap.get(rnum);
%>
		<div class="header-contents vitaminb" style="background-color : <%=backcol%>">

			<ul id="item-listA" class="customer-review-list-wrap clearfix">
				<li class="has-image">
					<div class="contentA">
						<h1 style="width: 300px;">
							<span><%=vo.getRegion()%> | <%=vo.getCategory()%></span><br>
							<%=vo.getContent_name()%>
						</h1>
						<h4 class="only-sr" style="width: 300px;"><%=vo.getNewaddress()%></h4>
						<h4 class="only-sr" style="width: 300px;"><%=vo.getPhonenum()%></h4>
						<div class="btn-wrap">


							<%
								if (session.getAttribute("login-status") != null
											&& ((String) session.getAttribute("login-status")).equals("login")) {
							%>
							<button class="btn btn-radius btn-colored btn-shadow btn-cart"
								data="PDZBA94HTD2Y" data-title="비타민B"
								onclick ="window.location.href='/TeamProject/CustomInfo?action=plusmyplace&url=SearchPlace&myplaceid=<%=vo.getContent_code()%>'">여기 찜하기</button>
							<%
								} else {
							%>
							<button class="btn btn-radius btn-colored btn-shadow btn-cart"
								data="PDZBA94HTD2Y" data-title="비타민B" onclick="window.location.href='/TeamProject/jsp/Login.jsp'">여기 찜하기</button>
							<%
								}
							%>
						</div>
					</div>
				</li>
				<li class="has-image">
					<div>
						<div id="map" style="width: 800px; height: 400px; margin-left: 53px;"></div>
						<script type="text/javascript">
function Mapadapt(y,x){
	var mapContainer = document.getElementById('map'), // 지도를 표시할 div 
	mapOption = {
		center : new daum.maps.LatLng(<%=vo.getY()%>,<%=vo.getX()%>), // 지도의 중심좌표
		level : 3
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
Mapadapt(<%=vo.getY()%>,<%=vo.getX()%>);
</script>
					</div>
				</li>
			</ul>
		</div>

		<main id="main" class="product-detail-page">
		<div>
			<section class="customer-review wrapper">
				<h3 class="section-title">고객 후기</h3>
				<ul id="item-list" class="customer-review-list-wrap clearfix">
					<li class="has-image"
						data-id="76cca5ef-188f-11e9-bcdf-0a75473d8c48">
						<div class="review-image"
							style="background-image: url(https://img.pilly.kr/review/201901/20190115/7269ad5d-72d5-41b7-ba05-e0e24a8ae315.jpg)">
							<span class="badge-BEST"></span>
						</div>
						<div class="open-contents">
							<p>육아에 지쳐서인지, 아니면 나이가 먹어서인지.. 요즘들어 몸은 피곤하고 면역력은 떨어져혓바늘도 나고,
								구내염도 나고 ㅠㅠ피곤한 나날들을 보낸다고 블로그에 포스팅하니, 많은 분들이 비타민이나 영양제를 추천하시더라구요사실
								그동안 젊다는 이유로 챙겨먹지 않았었는데.. 이제 본격적으로 나도 영양제를 먹어야하나 싶었어요근데 비타민이고
								영양제고 워낙 종류가 많다보니, 내 몸에 딱 맞는 맞춤영양제를 챙기고 싶더라구요 그러다가 알게 된 게 바로 이
								필리영양제에요! 필리는 홈페이지에 접속하면 내 몸에 딱 맞는 맞춤영양제를 진단해줌은 물론,배송하고</p>
						</div>
						<div class="open-more link-info"
							data-id="76cca5ef-188f-11e9-bcdf-0a75473d8c48">&gt;&gt;
							블로그에서보기</div>
					</li>
					<li class="has-image"
						data-id="cc69d3ca-e31b-4c74-9518-ce0a36c87174">
						<div class="review-image"
							style="background-image: url(https://img.pilly.kr/review/201812/20181203/93aa4ab9-c2d4-4121-a3ae-7e13f069f601.GIF)"></div>
						<div class="open-contents">
							<p>요즘 아이 둘 육아하며 일하느라 만성피로를 얻고 안아픈 곳이 없는 요즘.. 사실 아이들 돌보느라 영양제
								챙겨먹을 시간도, 그리고 까먹는게 다반사에요. 그래서 고민하던 차에, 주문 전 개인 문진을 통해 본인에게 맞는
								맞춤형 영양제를 추천 받고, 필리케어(카카오톡친구)를 통해 영양제 섭취를 관리해준다고 하는 필리 영양제를
								주문해보았어요. 1. 홈페이지를 통해 개인 문진을 해보아요. 필리 : 개인 맞춤형 영양제 정기구독 서비스 내게 맞는
								올바른 영양제를 찾아 편하게 섭취하자. pilly.kr 모바일로도</p>
						</div>
						<div class="open-more link-info"
							data-id="cc69d3ca-e31b-4c74-9518-ce0a36c87174">&gt;&gt;
							블로그에서보기</div>
					</li>
					<li class="has-image"
						data-id="1ca2978d-c57a-4dca-8b55-5d7704deac6f">
						<div class="review-image"
							style="background-image: url(https://img.pilly.kr/review/201811/20181130/71feb050-0087-4c9d-8a27-ce8832d075e6.jpg)"></div>
						<div class="open-contents">
							<p>오늘은 필리 영양제에 대한 후기를 써보려고 합니다! 20살에서 고작 한 살 먹었다고.. 물론 젊은
								나이지만... 급격히 체력이 딸리는 걸 느끼고 시험 기간이기도 하고 ㅜ.ㅜ 항상 영양제 사서 먹어야지 ~~ 생각을
								하고 있었는데요! 마침 지인이 영양제를 보내줄까? 해서 덥석 받아 보게 되었어용 크크 고것은 바로바로 필리
								영양제인데요! 필리 홈페이지에 들어가면 간단한 테스트를 할 수 있고, 테스트 후에는 자신에게 맞는 영양제 리스트들이
								나와요! 저는 루테인, 오메가3, 칼마디, 비타민씨, 밀크씨슬, 비타민비</p>
						</div>
						<div class="open-more link-info"
							data-id="1ca2978d-c57a-4dca-8b55-5d7704deac6f">&gt;&gt;
							블로그에서보기</div>
					</li>
				</ul>
			</section>
		</div>
	</div>
	</main>
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
	</div>
	<%
		}
	%>
</body>
</html>