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
<script src="//code.jquery.com/jquery-1.11.0.min.js"></script>
<script
	src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
<title>내일 미세</title>
<style type="text/css">
.top_photo_area_wrap {
	position: relative;
	max-width: auto;
	height: 312px;
	margin: -1px auto;
	background: #e2e2e2;
}
#content {
	width: 900px;
	margin: -1px auto 0;
	padding: 30px;
	border: 1px solid #f3f3f3;
	background: #fff;
	-webkit-box-sizing: border-box;
	box-sizing: border-box;
	-webkit-box-shadow: 0 0 6px rgba(0, 0, 0, 0.02);
	box-shadow: 0 0 6px rgba(0, 0, 0, 0.02);
}

.ct_box_area {
	position: relative;
	z-index: 2010;
	margin-top: -82px;
	padding-bottom: 24px;
	border: 1px solid #f0f0f0;
	background: #fff;
	-webkit-box-shadow: 0 0 6px rgba(0, 0, 0, 0.03);
	box-shadow: 0 0 6px rgba(0, 0, 0, 0.03);
}

.biz_name_area {
	padding: 29px 156px 18px;
	text-align: center;
	border-top: 7px solid;
	border-color: #5cb85c;
}

.name {
	margin-right: 10px;
	font-size: 32px;
	font-weight: bold;
	color: #000;
	line-height: 40px;
}

strong, b {
	font-weight: 700;
}

.category {
	font-size: 16px;
	color: #9a9a9a;
	line-height: 20px;
}

.list_item_menu {
	position: relative;
}

.list_item {
	margin-top: 20px;
	margin-bottom: 20px;
	line-height: 25px;
}

.txt {
	overflow: hidden;
	color: #424242;
	font-size: 20px;
}

.tit {
	float: left;
	margin-top: 4px;
	margin-right: 12px;
	font-size: 20px;
}
.menu, .worktime, .price {
	font-size: 20px;
}
</style>
</head>
<body>
	<%
		ArrayList<RecommendVO> list = (ArrayList<RecommendVO>) session.getAttribute("selectfive");
	%>
	<%
		if (request.getParameter("Menu_num") != null) {
			RecommendVO vo = list.get(Integer.parseInt(request.getParameter("Menu_num")));
	%>

	<ul class="survey-result-list-wrap">
		<div id="container" role="main">
			<div class="top_photo_area_wrap"
				style="background-image: url(<%=vo.getPicture1()%>)"></div>

			<div id="content" class="content">
				<div class="ct_box_area">
					<div class="biz_name_area">
						<strong class="name"><%=vo.getPlace()%></strong><span
							class="category"><%=vo.getCate_f()%></span>
					</div>
				</div>


				<div class="list_item list_item_desc">
					<span class="tit" aria-label="업체설명" role="img"><svg
							class="icon" role="presentation" version="1.1" width="20"
							height="20" viewBox="0 0 17 19">
								<path fill="#9E9E9E"
								d="M13,8V0H0v17h12h1h2c1.104,0,2-0.895,2-2V8H13z M12,8v8H1V1h11V8z M16,14.413c0,0.884-0.716,1.6-1.6,1.6H13 V9h3V14.413z M10,8H3v1h7V8z M7,11H3v1h4V11z M10,5H3v1h7V5z"></path></svg></span>
					<div class="txt">
						<div class="ellipsis_area">
							<%
								if (vo.getPrice().equals("None")) {
										vo.setPrice("");
									}
							%>
							<span class="price"><%=vo.getPrice()%></span>
						</div>
					</div>
				</div>


				<div class="list_item list_item_menu">
					<span class="tit" aria-label="메뉴" role="img"><svg
							class="icon" role="presentation" version="1.1" width="20"
							height="20" viewBox="0 0 17 19">
								<path fill="#9E9E9E"
								d="M4.5,0C2.567,0,1,1.567,1,3.5v3c0,1.762,1.306,3.205,3,3.45V17h1V9.95C6.694,9.705,8,8.262,8,6.5v-3 C8,1.567,6.433,0,4.5,0z M6.8,6.5c0,1.268-1.032,2.3-2.3,2.3S2.2,7.769,2.2,6.5v-3c0-1.268,1.032-2.3,2.3-2.3s2.3,1.032,2.3,2.3V6.5 z M14.8,0v6.5c0,1.096-0.772,2.012-1.8,2.242V0h-1v8.742c-1.028-0.23-1.8-1.146-1.8-2.242V0H9v6.5c0,1.762,1.306,3.205,3,3.45V17h1 V9.95c1.694-0.245,3-1.688,3-3.45V0H14.8z"></path></svg></span>
					<div class="list_item list_item_menu">
						<div class="info menu">
							<ul class="list_menu">
								<li style="display: block">
									<div class="txt">
										<div class="menu_area">
											<%
												String menu = vo.getMenu();
													menu = menu.replaceAll("&", "<br>");
											%><span class="menu"><%=menu%></span>
										</div>
									</div>
								</li>
							</ul>
						</div>
					</div>
				</div>




				<div class="list_item list_item_biztime">
					<span class="tit" aria-label="영업시간" role="img"><svg
							class="icon" role="presentation" version="1.1" width="20"
							height="20" viewBox="0 0 17 19">
								<path fill="#9E9E9E"
								d="M11.114,12.118l-3.208-2.68V4.011h1.2v4.866l2.777,2.321L11.114,12.118z M17,8.5c0,4.694-3.806,8.5-8.5,8.5 	S0,13.194,0,8.5S3.806,0,8.5,0S17,3.806,17,8.5z M15.8,8.5c0-4.032-3.268-7.3-7.3-7.3c-4.032,0-7.3,3.269-7.3,7.3s3.268,7.3,7.3,7.3 	C12.532,15.8,15.8,12.532,15.8,8.5z"></path></svg></span>
					<div class="txt">
						<div class="biztime_area list_more_view" data-page="detailCommon"
							data-action="bizhour" target="_self">
							<div class="biztime_row ">
								<div class="biztime ">
									<span class="time ">
									<%
												String worktime = vo.getWorktime();
									worktime = worktime.replaceAll("&", "<br>");
											%>
									<span class=worktime><%=worktime%></span></span>
								</div>
							</div>
						</div>
					</div>
				</div>




				<div class="list_item list_item_homepage">
					<span class="tit" aria-label="홈페이지" role="img"><svg
							class="icon" role="presentation" version="1.1" width="20"
							height="20" viewBox="0 0 17 19">
								<path fill="#9E9E9E"
								d="M15.375,9L15,8.598V15h-1V7.525l-5.5-5.9L3,7.525V15H2V8.598 L1.625,9H0l8.5-9L17,9H15.375z M11,9v6h-1V9H7v6H6V9V8h5V9z"></path></svg></span>
					<div class="txt">
						<a class="biz_url" href="http://www.facebook.com/Feeling8st"><%=vo.getHomepage()%></a>
					</div>
				</div>
			</div>
		</div>
	</ul>

	<%
		}
	%>
</body>
</html>
