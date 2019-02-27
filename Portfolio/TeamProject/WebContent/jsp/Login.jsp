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
<script
	src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>

<title>내일미세</title>
<style type="text/css">
body {
	background-color: #fafafa;
}

#login-view {
	position: absolute;
	left: 0;
	right: 0;
	top: 0;
	bottom: 0;
	width: 100%;
	background-color: #fafafa;
}

#login-view>div {
	display: table;
	width: 100%;
	height: 100%;
}

#login-view .login-view-inner {
	display: table-cell;
	text-align: center;
	vertical-align: middle;
	padding-top: 100px;
}

a {
	text-decoration: none;
	color: inherit;
}

section {
	margin: auto;
	width: 458px;
}

.mise-login {
	margin-top: 50px;
}

.Mainicon {
	height: 100px;
}

.input-full {
	width: 100%;
}

.input-text {
	padding: 0 20px;
	height: 60px;
	font-size: 20px;
	border: 1px solid #d7d7d7;
	border-radius: 4px;
	outline: none;
	background-color: #fff;
	box-sizing: border-box;
}

.grid-full {
	margin-top: 20px;
}

.btn-login {
	margin: 32px 0 26px;
}

.btn.btn-colored {
	line-height: 50px;
}

.btn-success {
	border: none;
	color: #fff;
}

.btn-full {
	width: 100%;
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
	vertical-align: middle;
	box-sizing: border-box;
}

.btn-colored {
	background-color: #00b050;
	color: #ffffff;
}

input {
	cursor: pointer;
}

a {
	text-decoration: none;
	color: rgba(51, 51, 51, 0.6);
	font-size: 16px;
	letter-spacing: -1px;
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
	<div id="login-view">
		<div>
			<div class="login-view-inner">
				<h1 class="site-logo-login">
					<a href="/"><img src="images/basiclogo.png" class="Mainicon"></a>
				</h1>
				<section class="mise-login">
					<form method="Post" action="/TeamProject/CustomInfo" id="login-form"
						class="form" >
						<input type="text" class="input-text input-full" name="id"
							placeholder="아이디(이메일)을 입력해 주세요." required="required"
							data-parsley-required-message="아이디를 입력해 주세요."
							data-parsley-errors-container="#login-result"> <br>
						<br> <input type="password" name="password"
							class="input-text input-full" placeholder="비밀번호를 입력해 주세요."
							required="required" data-parsley-required-message="비밀번호를 입력해 주세요."
							data-parsley-minlength="4" data-parsley-maxlength="30"
							data-parsley-errors-container="#login-result">
						<p id="login-result" class="validation-error align-left"></p>
						<input type="hidden" name="action" value="login"> 
						<input	type="submit" value="로그인"	class="btn btn-full btn-radius btn-colored btn-login"> 
							<a	href="Join.jsp">회원가입</a>
					
					</form>
				</section>
			</div>
		</div>
	</div>
</body>
</html>