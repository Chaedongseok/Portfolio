<%@page import="Model.vo.CustomInfoVO"%>
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

#header {
	background: url("images/timesq.jpg");
	background-size: cover;
	background-repeat: no-repeat;
	height: 430px;
}

#header .header-contents {
	position: relative;
	padding-top: 180px;
	text-align: center;
}

.header-contents {
	margin-bottom: 26px;
	line-height: 42px;
	color: #fff;
	font-size: 40px;
	font-weight: medium;
	word-break: keep-all;
}

#main.place-page .place-filter {
	padding: 20px 0 17px;
	text-align: center;
	background-color: #fff;
	border-bottom: 4px solid rgba(0, 0, 0, 0.03);
}

.place-filter-list-wrap {
	display: inline-block;
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
	margin-block-start: 0em;
	margin-block-end: 0em;
	margin-inline-start: 0px;
	margin-inline-end: 0px;
	padding-inline-start: 0px;
}

li {
	display: list-item;
	text-align: -webkit-match-parent;
}

.place-filter-list-wrap li {
	float: left;
	width: 70px;
	font-size: 14px;
	font-weight: 700;
	text-align: center;
	white-space: nowrap;
	cursor: pointer;
	margin-left: 22px;
	margin-right: 20px;
}

svg {
	margin-left: auto;
	margin-right: 20px;
	fill: #333;
}

svg:hover {
	fill: #00B050;
}

}
.icon-all, .icon-bone, .icon-eye, .icon-heart, .icon-liver,
	.icon-antioxidant, .icon-immunity, .icon-energy {
	width: 50px;
	height: 50px;
}

.btn-icon {
	margin-top: -3px;
	padding: 0 4px 0 6px;
	vertical-align: middle;
	color: #333;
}

.icon {
	display: inline-block;
	width: 70px;
	height: 70px;
	background-repeat: no-repeat;
	background-position: 50% 50%;
}

h4 {
	font-size: 18px;
}

.h4, .h5, .h6, h4, h5, h6 {
	margin-top: 10px;
	margin-bottom: 10px;
}

#main.place-page .place-item-list-wrap {
	margin: 0 auto;
	padding-top: 12px;
	width: 1170px;
}

.place-item-list-wrap>li {
	float: left;
	margin: 54px 15px 0;
	width: 360px;
	border-radius: 0 0 6px 6px;
	box-sizing: border-box;
	box-shadow: 0px 2px 3px 0 rgba(0, 0, 0, 0.08);
}

.place-item-name {
	position: relative;
	height: 210px;
	border-radius: 6px 6px 0 0;
	background-repeat: no-repeat;
	background-position: 50% 50%;
	background-size: 88px;
	background-color: #8cc782;
}

.place-item-name>div {
	position: absolute;
	left: 30px;
	bottom: 32px;
	font-size: 20px;
	color: #333;
}

.place-item-name>div h3 {
	line-height: 1;
}

.place-item-name>div span {
	display: inline-block;
	margin-top: 9px;
	line-height: 1;
	font-size: 12px;
}

.place-item-info {
	padding: 24px 28px 30px;
	background-color: #fff;
}

.place-item-effect {
	padding-bottom: 22px;
	border-bottom: 1px solid #eee;
}

.place-item-effect>a {
	margin-left: 12px;
	font-size: 16px;
	font-weight: 700;
	color: #ccc;
	text-decoration: none;
}

.pointer {
	cursor: pointer;
}

.place-item-effect-icon li {
	float: left;
	margin-left: 20px;
	height: 50px;
	line-height: 50px;
}

.place-item-bottom {
	padding-top: 22px;
	text-align: center;
}

.icon.icon-plus {
	background-image: url(/images/icons/icon-plus.png);
	background-size: 14px;
	width: 14px;
	height: 14px;
}

.icon.icon-check-white {
	background-image: url(/images/icons/icon-check-white.png);
	background-size: 20px;
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

.btn-put-bucket {
	width: 243px;
	color: #848484;
	border: 2px solid #d7d7d7;
}
</style>
</head>
<body>

<script>
	var prevScrollpos = window.pageYOffset;
	window.onload = function() {
		window.onscroll = function() {
			var currentScrollPos = window.pageYOffset;
			if (prevScrollpos > currentScrollPos) {
				document.getElementById("navbar").style.background = "rgba(0, 255, 0, 0)";
				
				document.getElementById("menu-name2").style.color = "white";
	
				if (document.getElementById("menu-name4") != null) {
					document.getElementById("menu-name3").style.color = "white";
					document.getElementById("menu-name4").style.color = "white";
				}
				if (document.getElementById("menu-name5") != null) {
					document.getElementById("menu-name6").style.color = "white";
					document.getElementById("menu-name5").style.color = "white";
				}
	
			} else {
				document.getElementById("navbar").style.background = "white";
				
				document.getElementById("menu-name2").style.color = "black";
				if (document.getElementById("menu-name4") != null) {
					document.getElementById("menu-name3").style.color = "black";
					document.getElementById("menu-name4").style.color = "black";
				}
				if (document.getElementById("menu-name5") != null) {
					document.getElementById("menu-name6").style.color = "black";
					document.getElementById("menu-name5").style.color = "black";
				}
			}
			prevScrollpos = currentScrollPos;
		}
	}
	function getpcajax(region,category){
		  if (window.XMLHttpRequest) {
			    var xmlhttp=new XMLHttpRequest();
			  } else {  // code for IE6, IE5
			    var xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
			  }
			  xmlhttp.open("GET","/TeamProject/Recommend?region="+region+"&category="+category+"&action=searchplace",true);
			  xmlhttp.onreadystatechange=function() {
			    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
			      document.getElementById("show-card").innerHTML=xmlhttp.responseText;
			    }
			  }
			  xmlhttp.send();
	}
	function getpcajaxcat(category){
		  if (window.XMLHttpRequest) {
			    var xmlhttp=new XMLHttpRequest();
			  } else {  // code for IE6, IE5
			    var xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
			  }
			  xmlhttp.open("GET","/TeamProject/Recommend?category="+category+"&action=searchplaceregion",true);
			  xmlhttp.onreadystatechange=function() {
			    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
			      document.getElementById("show-card").innerHTML=xmlhttp.responseText;
			    }
			  }
			  xmlhttp.send();
	}
	function getpcajaxall(){
		  if (window.XMLHttpRequest) {
			    var xmlhttp=new XMLHttpRequest();
			  } else {  // code for IE6, IE5
			    var xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
			  }
			  xmlhttp.open("GET","/TeamProject/Recommend?action=searchplaceall",true);
			  xmlhttp.onreadystatechange=function() {
			    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
			      document.getElementById("show-card").innerHTML=xmlhttp.responseText;
			    }
			  }
			  xmlhttp.send();
	}
	getpcajaxall();
</script>

	<div>
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

		<main id="main" class="place-page">
		<nav class="place-filter">
			<div class="place-filter-list-wrap">
				<ul class="clearfix">
					<li data-target="all" data-target-text="전체" class="on btn-icon">
						<svg class="icon icon-all" xmlns="http://www.w3.org/2000/svg"
							xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
							y="0px" viewBox="0 0 48.001 60"
							enable-background="new 0 10 50 50" xml:space="preserve"
							onclick="getpcajaxall()">
					<polygon
								points="41.0005,44.5049 41.0005,48 48.0005,48 48.0005,44.5049 48.001,44.5 48.0005,44.4951 48.0005,41 41.0005,41   41.0005,44.4951 41,44.5 " />
					<circle cx="24" cy="44.5" r="3.5" />
					<circle cx="3.5" cy="44.5" r="3.5" />
					<circle cx="44.5" cy="24" r="3.5" />
					<polygon
								points="20.5005,24.0049 20.5005,27.5 27.5005,27.5 27.5005,24.0049 27.501,24 27.5005,23.9951 27.5005,20.5 20.5005,20.5   20.5005,23.9951 20.5,24 " />
					<circle cx="3.5" cy="24" r="3.5" />
					<circle cx="44.5" cy="3.5" r="3.5" />
					<circle cx="24" cy="3.5" r="3.5" />
					<polygon
								points="0.0005,0 0.0005,3.4951 0,3.5 0.0005,3.5049 0.0005,7 7.0005,7 7.0005,3.5049 7.001,3.5 7.0005,3.4951 7.0005,0 " />
					</svg><br>전체
					</li>
					<li data-target="bone" data-target-text="뼈" class="btn-icon"><svg
							class="icon icon-bone" xmlns="http://www.w3.org/2000/svg"
							data-name="Layer 1" viewBox="20 5 70 100" x="0px" y="0px"
							onclick="getpcajaxcat('백화점')">
					<path
								d="M60.56,9A19,19,0,0,0,39.44,9L14.87,25.46a1,1,0,0,0-.44.83V93.17a1,1,0,0,0,1,1H84.57a1,1,0,0,0,1-1V26.29a1,1,0,0,0-.44-.83ZM58.9,92.17H41.1V82.79a8.9,8.9,0,0,1,17.81,0Zm24.67,0H60.9V82.79a10.9,10.9,0,0,0-21.81,0v9.39H16.43V26.82L40.55,10.69a17,17,0,0,1,18.9,0L83.57,26.82Z" />
					<path
								d="M42.43,28.46H32.57a4,4,0,0,0-4,4v5.88a4,4,0,0,0,4,4h9.86a4,4,0,0,0,4-4V32.46A4,4,0,0,0,42.43,28.46Zm2,9.88a2,2,0,0,1-2,2H32.57a2,2,0,0,1-2-2V32.46a2,2,0,0,1,2-2h9.86a2,2,0,0,1,2,2Z" />
					<path
								d="M53.57,32.46v5.88a4,4,0,0,0,4,4h9.86a4,4,0,0,0,4-4V32.46a4,4,0,0,0-4-4H57.57A4,4,0,0,0,53.57,32.46Zm15.86,0v5.88a2,2,0,0,1-2,2H57.57a2,2,0,0,1-2-2V32.46a2,2,0,0,1,2-2h9.86A2,2,0,0,1,69.43,32.46Z" />
					<path
								d="M42.43,50.46H32.57a4,4,0,0,0-4,4v5.88a4,4,0,0,0,4,4h9.86a4,4,0,0,0,4-4V54.46A4,4,0,0,0,42.43,50.46Zm2,9.88a2,2,0,0,1-2,2H32.57a2,2,0,0,1-2-2V54.46a2,2,0,0,1,2-2h9.86a2,2,0,0,1,2,2Z" />
					<path
								d="M67.43,50.46H57.57a4,4,0,0,0-4,4v5.88a4,4,0,0,0,4,4h9.86a4,4,0,0,0,4-4V54.46A4,4,0,0,0,67.43,50.46Zm2,9.88a2,2,0,0,1-2,2H57.57a2,2,0,0,1-2-2V54.46a2,2,0,0,1,2-2h9.86a2,2,0,0,1,2,2Z" />
					</svg><br>백화점</li>
					<li data-target="eye" data-target-text="눈" class="btn-icon"><svg
							class="icon icon-eye" xmlns="http://www.w3.org/2000/svg"
							xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
							y="0px" viewBox="25 10 60 80"
							style="enable-background: new 0 0 100 100;" xml:space="preserve"
							onclick="getpcajaxcat('영화관')">
						<g>
						<path
								d="M83.5,50c0-18.74756-15.25195-34-34-34c-18.74756,0-34,15.25244-34,34c0,18.74805,15.25244,34,34,34h34v-2H60.9826   C74.0965,77.28003,83.5,64.7196,83.5,50z M17.5,50c0-17.64502,14.35498-32,32-32c17.64453,0,32,14.35498,32,32   c0,17.64453-14.35547,32-32,32C31.85498,82,17.5,67.64453,17.5,50z" />
						<path
								d="M53.5,50c0-2.20557-1.79395-4-4-4c-2.20557,0-4,1.79443-4,4c0,2.20605,1.79443,4,4,4C51.70605,54,53.5,52.20605,53.5,50z    M47.5,50c0-1.10303,0.89697-2,2-2s2,0.89697,2,2c0,1.10254-0.89697,2-2,2S47.5,51.10254,47.5,50z" />
						<path
								d="M49.5,24c-4.41113,0-8,3.58887-8,8s3.58887,8,8,8s8-3.58887,8-8S53.91113,24,49.5,24z M49.5,38c-3.30859,0-6-2.69141-6-6   s2.69141-6,6-6s6,2.69141,6,6S52.80859,38,49.5,38z" />
						<path
								d="M49.5,60c-4.41113,0-8,3.58887-8,8s3.58887,8,8,8s8-3.58887,8-8S53.91113,60,49.5,60z M49.5,74c-3.30859,0-6-2.69141-6-6   s2.69141-6,6-6s6,2.69141,6,6S52.80859,74,49.5,74z" />
						<path
								d="M75.5,50c0-4.41113-3.58887-8-8-8s-8,3.58887-8,8s3.58887,8,8,8S75.5,54.41113,75.5,50z M67.5,56c-3.30859,0-6-2.69141-6-6   s2.69141-6,6-6s6,2.69141,6,6S70.80859,56,67.5,56z" />
						<path
								d="M31.5,42c-4.41113,0-8,3.58887-8,8s3.58887,8,8,8s8-3.58887,8-8S35.91113,42,31.5,42z M31.5,56c-3.30859,0-6-2.69141-6-6   s2.69141-6,6-6s6,2.69141,6,6S34.80859,56,31.5,56z" /></g>
						</svg> <br>영화관</li>
					</svg>
					<li data-target="heart" data-target-text="혈액순환" class="btn-icon">
						<svg class="icon icon-heart" xmlns="http://www.w3.org/2000/svg"
							xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
							y="0px" viewBox="210 200 100 100"
							enable-background="190 170 100 100" xml:space="preserve"
							onclick="getpcajaxcat('극장')">
					<g>
					<path
								d="M300.533,238.857c-0.031-0.39-0.255-0.738-0.596-0.928c-0.342-0.19-0.756-0.198-1.104-0.019   c-0.12,0.062-12.195,6.177-26.358,6.18c-0.003,0-0.006-0.001-0.009-0.001c-2.087,0-4.246-0.135-6.438-0.4   c2.694-14.49,1.584-29.037,1.572-29.186c-0.031-0.39-0.255-0.738-0.596-0.928c-0.341-0.19-0.755-0.197-1.104-0.02   c-0.12,0.062-12.221,6.18-26.359,6.183c-0.003,0-0.007-0.001-0.011-0.001c-14.143,0-26.248-6.119-26.368-6.18   c-0.348-0.179-0.762-0.171-1.103,0.019c-0.341,0.19-0.565,0.539-0.596,0.928c-0.007,0.092-0.717,9.303,0.335,20.276   c1.425,14.855,5.226,25.721,11.298,32.295c4.378,4.74,9.907,7.145,16.434,7.145c0.001,0,0.003-0.001,0.005-0.001   c2.666,0,5.187-0.409,7.517-1.214c1.826,7.22,4.485,13.003,7.916,17.201c4.538,5.552,10.423,8.366,17.497,8.366   c0.002,0,0.004,0,0.005,0c6.524-0.002,12.051-2.405,16.428-7.144c6.072-6.575,9.873-17.44,11.299-32.296   C301.251,248.16,300.54,238.949,300.533,238.857z M224.819,265.492c-11.977-12.964-11.459-41.801-11.12-49.101   c3.828,1.707,14.011,5.681,25.823,5.683c0.004,0,0.008,0.001,0.012,0.001c11.819,0,22.009-3.979,25.837-5.686   c0.199,4.218,0.466,15.632-1.657,26.973c-10.04-1.615-17.539-5.414-17.615-5.454c-0.348-0.179-0.762-0.171-1.103,0.019   c-0.341,0.19-0.565,0.539-0.596,0.929c-0.014,0.173-0.308,3.969-0.232,9.644c-1.415-0.402-2.964-0.609-4.633-0.609   c-4.277,0-7.707,1.307-10.191,3.883c-4.351,4.511-4.16,11.242-4.149,11.526c0.018,0.496,0.347,0.927,0.821,1.074   c0.114,0.036,0.231,0.053,0.347,0.053c0.363,0,0.713-0.169,0.938-0.472c4.386-5.896,12.129-5.839,12.212-5.84   c0.016,0.001,0.031,0.001,0.047,0c0.025,0,2.341-0.033,5.169,0.911c0.397,4.167,0.998,8.09,1.79,11.69   c-2.146,0.772-4.491,1.163-6.983,1.163c-0.001,0-0.002,0-0.004,0C233.685,271.881,228.735,269.731,224.819,265.492z    M244.226,250.97c0.06,1.875,0.155,3.732,0.289,5.548c-2.645-0.739-4.716-0.743-4.979-0.739c-0.512-0.006-6.74,0.002-11.688,4.199   c0.392-2.015,1.261-4.592,3.179-6.581c2.027-2.103,4.89-3.168,8.509-3.168C241.272,250.23,242.845,250.48,244.226,250.97z    M287.183,289.843c-3.917,4.241-8.868,6.392-14.717,6.392c-0.001,0-0.002,0.001-0.004,0.001   c-11.674-0.003-19.563-8.434-23.448-25.059c-0.928-3.971-1.611-8.38-2.03-13.103c-0.227-2.568-0.377-5.24-0.448-7.942   c-0.104-4.131,0.002-7.49,0.094-9.391c2.923,1.3,9.572,3.929,17.853,5.103c2.712,0.386,5.395,0.582,7.973,0.582   c0.004,0,0.007,0.001,0.011,0.001c11.817,0,22.006-3.977,25.835-5.684C298.642,248.041,299.164,276.87,287.183,289.843z" />
					<path
								d="M285.987,275.195c-0.474-0.146-0.989,0.021-1.285,0.419c-4.222,5.677-11.376,5.84-12.178,5.84h-0.034   c-0.015,0-0.031,0-0.045,0h-0.036c-0.803,0-7.956-0.163-12.177-5.84c-0.221-0.297-0.568-0.472-0.938-0.472   c-0.12,0-1.171,0.039-1.171,1.366c0,1.038,0.14,4.697,1.943,8.18c1.652,3.189,5.145,6.99,12.4,6.99   c4.277,0,7.706-1.307,10.19-3.883c4.352-4.511,4.161-11.242,4.15-11.526C286.789,275.773,286.46,275.343,285.987,275.195z    M280.975,286.173c-2.027,2.103-4.891,3.168-8.509,3.168c-3.612,0-6.471-1.062-8.498-3.156c-1.917-1.981-2.789-4.568-3.184-6.591   c4.806,4.062,10.842,4.198,11.624,4.198h0.058h0.058c0.782,0,6.824-0.137,11.631-4.204   C283.763,281.603,282.895,284.182,280.975,286.173z" />
					<path
								d="M265.508,261.123c0,0.646,0.522,1.169,1.168,1.169c0.646,0,1.169-0.523,1.169-1.169c0-4.446-3.617-8.063-8.063-8.063   c-4.446,0-8.063,3.617-8.063,8.063c0,0.646,0.523,1.169,1.169,1.169s1.169-0.523,1.169-1.169c0-3.157,2.568-5.726,5.726-5.726   C262.939,255.397,265.508,257.966,265.508,261.123z" />
					<path
								d="M285.149,253.06c-4.445,0-8.062,3.617-8.062,8.063c0,0.646,0.522,1.169,1.168,1.169s1.169-0.523,1.169-1.169   c0-3.157,2.568-5.726,5.725-5.726c3.157,0,5.727,2.568,5.727,5.726c0,0.646,0.523,1.169,1.169,1.169s1.169-0.523,1.169-1.169   C293.213,256.677,289.596,253.06,285.149,253.06z" />
					<path
								d="M252.217,237.939c4.446,0,8.063-3.617,8.063-8.063c0-0.645-0.523-1.168-1.169-1.168s-1.169,0.523-1.169,1.168   c0,3.157-2.568,5.726-5.726,5.726c-3.157,0-5.725-2.568-5.725-5.726c0-0.645-0.523-1.168-1.168-1.168   c-0.646,0-1.169,0.523-1.169,1.168C244.154,234.322,247.771,237.939,252.217,237.939z" />
					<path
								d="M234.913,229.875c0-0.645-0.523-1.168-1.169-1.168s-1.169,0.523-1.169,1.168c0,3.157-2.568,5.726-5.725,5.726   c-3.157,0-5.726-2.568-5.726-5.726c0-0.645-0.523-1.168-1.169-1.168c-0.645,0-1.168,0.523-1.168,1.168   c0,4.446,3.617,8.063,8.063,8.063C231.296,237.939,234.913,234.322,234.913,229.875z" /></g></svg>
						<br>극장
					</li>
					<li data-target="liver" data-target-text="간" class="btn-icon">
						<svg class="icon icon-liver" xmlns="http://www.w3.org/2000/svg"
							xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
							y="0px" viewBox="0 0 60 78" enable-background="new 0 0 68 68"
							xml:space="preserve"
							onclick="getpcajaxcat('공연장')">
					<g>
					<path
								d="M60.5,1.3C60,0.9,59.3,0.9,59,1.4c-0.6,0.8-1.2,1.6-1.8,2.4l-1.9-1.9c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l2,2   c-0.6,0.7-1.2,1.3-1.9,2l-2.1-2.1c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l2,2c-0.6,0.6-1.2,1.1-1.9,1.7c-1.1,1-1.8,2.3-1.8,3.8   L34,29l-0.6-0.6c-4-4-10.7-2.6-13.4,4.5c-0.6,1.7-2.3,2.9-4.1,3c-4.2,0.4-8.2,2.1-11,4.9c-5.7,5.7-5.1,14.5,1.4,21   c6.1,6.1,14.9,7.4,21,1.4c2.8-2.8,4.6-6.8,4.9-11c0.1-1.9,1.3-3.5,3-4.1c7.2-2.7,8.2-9.6,4.4-13.4L39,34l14.8-14.8c0,0,0,0,0.1,0   c1.4,0,2.8-0.6,3.7-1.7c0.6-0.7,1.2-1.3,1.8-1.9l2,2c0.4,0.4,1,0.4,1.4,0c0.4-0.4,0.4-1,0-1.4l-2-2c0.7-0.7,1.3-1.3,2-1.9l1.9,1.9   c0.4,0.4,1,0.4,1.4,0c0.4-0.4,0.4-1,0-1.4l-1.8-1.8c0.8-0.7,1.6-1.3,2.3-1.9C66.8,8.8,67,8.6,67,8.3c0-0.3-0.1-0.6-0.3-0.8   L60.5,1.3z M56,16.2c-0.6,0.7-1.6,1.1-2.6,1c-0.3,0-0.6,0.1-0.8,0.3L36.9,33.3c-0.2,0.2-0.3,0.4-0.3,0.7c0,0.3,0.1,0.5,0.3,0.7   l1.3,1.3c2.8,2.8,2.2,7.9-3.7,10.2c-2.4,0.9-4.1,3.2-4.3,5.8c-0.3,3.8-1.9,7.3-4.3,9.7c-5.2,5.2-12.8,3.9-18.1-1.4   C2,54.7,1.5,47,6.3,42.2c2.4-2.4,6-4,9.7-4.3c2.6-0.2,4.9-1.9,5.8-4.3c2.2-5.7,7.1-6.7,10.1-3.8l1.4,1.4c0.4,0.4,1,0.4,1.4,0   l15.8-15.8c0.2-0.2,0.3-0.5,0.3-0.8c-0.1-1,0.3-1.9,1-2.6c3.3-2.8,5.9-5.6,8-8.4l4.6,4.6C61.4,10.6,58.6,13.3,56,16.2z" />
					<path
								d="M9.7,48.2c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l10.1,10.1c0.4,0.4,1,0.4,1.4,0s0.4-1,0-1.4L9.7,48.2z" />
					<path
								d="M13.8,44.1c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l10.1,10.1c0.4,0.4,1,0.4,1.4,0c0.4-0.4,0.4-1,0-1.4L13.8,44.1z" />
					<path
								d="M24.5,36.5c-1.9,1.9-1.9,5,0,6.9c1.9,1.9,5,1.9,6.9,0c1.9-1.9,1.9-5,0-6.9C29.6,34.6,26.5,34.6,24.5,36.5z M30,42   c-1.1,1.1-3,1.1-4.1,0c-1.1-1.1-1.1-3,0-4.1c1.1-1.1,3-1.1,4.1,0C31.2,39.1,31.2,40.9,30,42z" /></g>
					</svg> <br>공연장
					</li>
					<li data-target="antioxidant" data-target-text="항산화"
						class="btn-icon"><svg class="icon icon-antioxidant"
							xmlns="http://www.w3.org/2000/svg" data-name="Capa 1"
							viewBox="0 0 20 20" x="0px" y="0px" onclick="getpcajaxcat('박물관')">
					<path class="cls-1"
								d="M18.88,15.89H18.09a1.09,1.09,0,0,0-.89-0.9V7.85h0.1A0.84,0.84,0,0,0,18.14,7V6.91h0.92V6.73a1.48,1.48,0,0,0-1-1.41L10.93,2.95A2.8,2.8,0,0,0,10,2.8H10a2.8,2.8,0,0,0-.89.14L1.95,5.32a1.48,1.48,0,0,0-1,1.41V6.91H1.87V7a0.84,0.84,0,0,0,.84.84H2.8V15a1.09,1.09,0,0,0-.9.9H1.12A1.12,1.12,0,0,0,0,17V17.2H20V17A1.12,1.12,0,0,0,18.88,15.89Zm-2-.93h-1.5V7.85h1.5V15ZM5.06,7.85V6.91H5.15V7a0.83,0.83,0,0,0,.47.75V15H5.05V7.85h0Zm2.44-.37H6A0.47,0.47,0,0,1,5.52,7V6.91H8V7A0.47,0.47,0,0,1,7.49,7.48ZM6,7.85h1.5V15H6V7.85Zm1.87-.09A0.83,0.83,0,0,0,8.33,7V6.91H8.41V7a0.83,0.83,0,0,0,.47.75V15h-1V7.76Zm2.89-.28H9.25A0.47,0.47,0,0,1,8.78,7V6.91h2.45V7A0.47,0.47,0,0,1,10.75,7.48Zm-1.5.37h1.5V15H9.25V7.85Zm1.87-.09A0.83,0.83,0,0,0,11.59,7V6.91h0.09V7a0.83,0.83,0,0,0,.47.75V15h-1V7.76ZM14,7.48H12.52A0.47,0.47,0,0,1,12,7V6.91H14.5V7A0.47,0.47,0,0,1,14,7.48Zm-1.5.37H14V15H12.52V7.85Zm1.88-.1A0.83,0.83,0,0,0,14.86,7V6.91H15V7.85h0V15H14.4V7.75ZM17.77,7a0.47,0.47,0,0,1-.47.47h-2V6.91h2.45V7ZM1.32,6.54a1.11,1.11,0,0,1,.75-0.87L9.19,3.3A2.43,2.43,0,0,1,10,3.17H10a2.43,2.43,0,0,1,.77.12l7.12,2.37a1.11,1.11,0,0,1,.75.87H1.32ZM2.71,7.48A0.47,0.47,0,0,1,2.24,7V6.91H4.69V7.48h-2Zm2,0.37V15H3.17V7.85H4.69ZM3,15.33H17a0.75,0.75,0,0,1,.73.57H2.26A0.75,0.75,0,0,1,3,15.33Zm-2.6,1.5a0.75,0.75,0,0,1,.73-0.57H18.88a0.75,0.75,0,0,1,.73.57H0.39Z"></path></svg>
						<br>박물관</li>
					<li data-target="immunity" data-target-text="면역" class="btn-icon">
						<svg class="icon icon-immunity"
							xmlns:dc="http://purl.org/dc/elements/1.1/"
							xmlns:cc="http://creativecommons.org/ns#"
							xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
							xmlns:svg="http://www.w3.org/2000/svg"
							xmlns="http://www.w3.org/2000/svg"
							xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
							xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
							onclick="getpcajaxcat('미술관')"
							version="1.1" x="0px" y="0px" viewBox="8 5 80 80">
					<g transform="translate(0,-952.36218)">
					<path
								style="font-size:medium;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;text-indent:0;text-align:start;text-decoration:none;line-height:normal;letter-spacing:normal;word-spacing:normal;text-transform:none;direction:ltr;block-progression:tb;writing-mode:lr-tb;text-anchor:start;baseline-shift:baseline;opacity:1;color:#000000;stroke:none;stroke-width:1.99999869000000000;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate;font-family:Sans;-inkscape-font-specification:Sans"
								d="M 29 18 L 29 20 L 31 20 L 31 23 L 19 23 L 18 23 L 18 24 L 18 68 L 18 69 L 19 69 L 58.6875 69 C 59.544135 73.039056 61.256689 76.760241 64.15625 81.5 L 64.4375 82 L 65 82 L 75 82 L 75.5625 82 L 75.84375 81.5 C 80.302878 74.2111 82 69.3148 82 62 L 82 61.96875 C 81.94118 58.66125 80.871599 56.0416 79.875 53.8125 C 78.878401 51.5834 78 49.7619 78 48 C 78 46.45107 78.230966 45.24947 80.53125 43.84375 L 81 43.5625 L 81 43 L 81 37 L 81 36 L 80 36 L 74 36 L 74 24 L 74 23 L 73 23 L 61 23 L 61 20 L 63 20 L 63 18 L 57 18 L 57 20 L 59 20 L 59 23 L 33 23 L 33 20 L 35 20 L 35 18 L 29 18 z M 20 25 L 72 25 L 72 36 L 69 36 L 69 29 L 69 28 L 68 28 L 24 28 L 23 28 L 23 29 L 23 63 L 23 64 L 24 64 L 58.0625 64 C 58.111705 65.032781 58.210495 66.03913 58.34375 67 L 20 67 L 20 25 z M 25 30 L 67 30 L 67 36 L 60 36 L 59 36 L 59 37 L 59 43 L 59 43.5625 L 59.46875 43.84375 C 61.76903 45.24947 62 46.45121 62 48 C 62 49.7619 61.121598 51.5834 60.125 53.8125 C 59.128402 56.0416 58.058818 58.66125 58 61.96875 L 58 62 L 25 62 L 25 30 z M 61 38 L 79 38 L 79 42.5 C 76.679397 44.14413 76 46.22405 76 48 C 76 50.3869 77.055432 52.4424 78.03125 54.625 C 79.007068 56.8076 79.948487 59.1034 80 62 C 80 68.8543 78.542739 73.1494 74.40625 80 L 65.59375 80 C 61.457261 73.1494 60 68.8543 60 62 C 60.05151 59.1034 60.992931 56.8076 61.96875 54.625 C 62.944569 52.4424 64 50.387 64 48 C 64 46.22416 63.320606 44.14414 61 42.5 L 61 38 z "
								transform="translate(0,952.36218)" 
								/></g>
					</svg><br>미술관
					</li>
					<li data-target="energy" data-target-text="에너지" class="btn-icon">
						<svg class="icon icon-energy" xmlns="http://www.w3.org/2000/svg"
							xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px"
							y="0px" viewBox="10 5 90 80"
							style="enable-background: new 0 0 100 100;" xml:space="preserve"
							onclick="getpcajaxcat('키즈카페')">
					<g>
					<path
								d="M88.2,44c-4.3,0.4-8.6,1.2-12.8-0.3c-2.9-1-5.7-2.7-8.4-0.3c-0.3,0.2-0.5,0.6-0.6,1.1c-4.6,1.7-8.4,5.5-9.8,10.7   c-2.2,8.3,2.7,16.9,11.1,19.1c8.3,2.2,16.9-2.7,19.1-11.1c1.1-4.2,0.4-8.6-1.7-12.1c1.2-0.8,2.2-1.8,2.9-2.7   c0.7-0.8,1.7-2.1,1.9-3.3C90.2,43.7,89.9,43.9,88.2,44z M84.8,62.7c-1.9,7.2-9.4,11.5-16.6,9.6c-7.2-1.9-11.5-9.4-9.6-16.6   c1.1-4.2,4.2-7.5,7.9-9c0.2,0.6,0.6,1.3,1.1,1.9c1.9,2.3,5.5,4.6,10.1,4.7c2,0.1,3.9-0.4,5.5-1.1C85.2,55.2,85.8,59,84.8,62.7z" />
					<path
								d="M74.7,64.9c-2.9-0.2-5.7-0.9-8.3-2.2c-1-0.5-2.3,0.2-2.2,1.4c0,2,1.8,4.3,4.9,5.2c3.2,0.8,5.8-0.2,6.9-2   C76.5,66.3,75.8,65,74.7,64.9z" />
					<ellipse
								transform="matrix(0.2588 -0.9659 0.9659 0.2588 1.3796 119.5178)"
								cx="78.6" cy="58.9" rx="2.1" ry="2.1" />
					<ellipse
								transform="matrix(0.2588 -0.9659 0.9659 0.2588 -4.6881 104.8689)"
								cx="66" cy="55.5" rx="2.1" ry="2.1" />
					<path
								d="M47.4,47.6c-2.6,1.3-5.4,2-8.3,2.2c-1.2,0.1-1.8,1.3-1.2,2.3c1.1,1.7,3.7,2.8,6.9,2c3.2-0.9,4.9-3.1,4.9-5.2   C49.7,47.8,48.5,47.1,47.4,47.6z" />
					<ellipse
								transform="matrix(0.9659 -0.2588 0.2588 0.9659 -8.8225 13.7467)"
								cx="47.8" cy="40.4" rx="2.1" ry="2.1" />
					<ellipse
								transform="matrix(0.9659 -0.2588 0.2588 0.9659 -10.1235 10.6058)"
								cx="35.2" cy="43.8" rx="2.1" ry="2.1" />
					<path
								d="M57.1,40C54.9,31.7,46.3,26.7,38,29c-0.3,0.1-0.6,0.2-0.9,0.3c0,0.1,0,0.2,0,0.3c0,0,0,0,0,0c0,0,0,0.1,0,0.1c0,0,0,0,0,0   c0,0,0,0.1,0,0.1c-0.2,0.6-0.6,1.1-1.2,1.3l-3.9,1.2l-0.4,0.1c-0.1,0.1-0.1,0.1-0.2,0.2l-1.5,4.3c-0.2,0.6-0.7,1-1.2,1.1   c0,0-0.1,0-0.1,0c0,0,0,0,0,0c0,0-0.1,0-0.1,0c0,0,0,0,0,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0   c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0-0.1,0c0,0,0,0,0,0c0,0-0.1,0-0.1,0c-1.3,3.1-1.6,6.6-0.7,10   c2.2,8.3,10.8,13.3,19.1,11.1C54.4,56.9,59.4,48.4,57.1,40z M45.5,57.2c-7.2,1.9-14.7-2.4-16.6-9.6c-0.7-2.5-0.6-5,0.1-7.4   l11.6-3.1l1.2-5.5l3.8,4.2l5.7-1.5c1.8,1.7,3.1,3.8,3.8,6.3C57,47.8,52.7,55.2,45.5,57.2z" />
					<path
								d="M24.2,33.4l3.4,3.5c0.4,0.4,1.1,0.3,1.3-0.3l1.7-4.9l4.9-1.5c0.5-0.2,0.7-0.8,0.3-1.2l-3.4-3.5c-0.4-0.4-1.1-0.3-1.3,0.3   l-1.7,4.9l-4.9,1.5C24,32.3,23.8,33,24.2,33.4z" />
					<path
								d="M26.5,37.3l-3.1-3.2C23,33.6,22.8,33,23,32.4c0.2-0.6,0.6-1.1,1.2-1.3l4.5-1.4l0.1-0.2c-0.8-0.5-2-1-3.1-0.8   c-2,0.2-3.9,2.3-5.5,4.7c-1.7,2.5-3.4,4.1-5.2,4.2c-1.8,0.2-3.8-1.4-4.6-1.4c-0.4,0-0.3,0.7,0.2,2c0.5,1.3,1.6,3.2,3,5.1   c1.4,1.8,3.1,3.3,4.7,3.5c1.6,0.2,3.2-1,4.6-3.2C24.3,41.3,25.5,39,26.5,37.3C26.5,37.3,26.5,37.3,26.5,37.3z" /></g></svg><br>키즈카페
					</li>
				</ul>
			</div>
		</nav>

		<div id= "show-card">
			<h1>show card</h1>		
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