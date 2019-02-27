<%@page import="java.util.HashMap"%>
<%@page import="Model.vo.SearchPlaceVO"%>
<%@page import="java.util.ArrayList"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
<% if (session.getAttribute("searchplace")!=null){ 
	ArrayList<SearchPlaceVO> list = (ArrayList<SearchPlaceVO>)session.getAttribute("searchplace");
	HashMap<Integer,String> colormap= new HashMap<Integer,String>();
	colormap.put(1,"#8cc782");
	colormap.put(2,"#a7a8d3");
	colormap.put(3,"#ead499");
	colormap.put(4,"#d3d1d1");
	colormap.put(5,"#81b0ce");
	colormap.put(6,"#e6aa7c");
	int cnt=1;
%>
	<section>
		
		<ul class="place-item-list-wrap clearfix">
			<%for(SearchPlaceVO vo :list){ %>
				<li>
					<section>
						<div class="place-item-name link-info pointer" data="PDZBA94HTD2Y"
							data-nick="vitaminb"
							style="background-image: url(https://img.pilly.kr/place/tablet/vitaminb.png); background-color:<%=colormap.get(cnt)%>;">
							<div>
								<h3><%=vo.getContent_name() %></h3>
								<span><%=vo.getRegion() %>  |  <%=vo.getCategory() %></span>
							</div>
						</div>
						<div class="place-item-info">
							<section class="place-item-effect link-info pointer"
								data="PDZBA94HTD2Y" data-nick="vitaminb">
								<h4 class="only-sr"><%=vo.getNewaddress() %></h4>
								<h4 class="only-sr"><%=vo.getPhonenum() %></h4>
								<a href="/TeamProject/Recommend?action=showdetail&content-code=<%=vo.getContent_code()%>">상세보기</a>
							</section>
							<div class="place-item-bottom">
								<div class="btn-put-bucket-wrap">
								<%if(session.getAttribute("login-status")!=null&&
								((String)session.getAttribute("login-status")).equals("login")){ %>
									<button class="btn btn-radius btn-put-bucket"
										data="PDZBA94HTD2Y" data-title="비타민B"
										onclick ="window.location.href='/TeamProject/CustomInfo?action=plusmyplace&url=SearchPlace&myplaceid=<%=vo.getContent_code()%>'">
										<span>여기 찜하기</span>
									</button>
								<%}else{ %>
									<button class="btn btn-radius btn-put-bucket"
										data="PDZBA94HTD2Y" data-title="비타민B"
										onclick ="window.location.href='/TeamProject/jsp/Login.jsp'">
										<span>여기 찜하기</span>
									</button>
								<%} %>
								</div>
							</div>
						</div>
					</section>
				</li>
				<% if(cnt>5) cnt=1;
				else cnt++;%>
			<%} %>

		</ul>
	</section>
<%}else{%>
<h1>실패!</h1>
<%} %>
</body>
</html>