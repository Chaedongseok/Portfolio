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

</style>

</head>
<body>
<script>
function getSurvey(question_num) {/* Survey_content.jsp에서 내용물을 끄집어와 Ajax로 보여줌 */
  if (window.XMLHttpRequest) {
    var xmlhttp=new XMLHttpRequest();
  } else {  // code for IE6, IE5
    var xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
  }
  xmlhttp.open("GET","/TeamProject/jsp/SurveyContent.jsp?question_num="+question_num,true);
  xmlhttp.onreadystatechange=function() {
    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
      document.getElementById("Survey_show").innerHTML=xmlhttp.responseText;
    }
  }
  xmlhttp.send();
}
function getResult(question_num,region) {/* Survey_content.jsp에서 내용물을 끄집어와 Ajax로 보여줌 */
	  if (window.XMLHttpRequest) {
	    var xmlhttp=new XMLHttpRequest();
	  } else {  // code for IE6, IE5
	    var xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	  }
	  xmlhttp.open("GET","/TeamProject/jsp/SurveyContent.jsp?question_num="+question_num+"&region="+region,true);
	  xmlhttp.onreadystatechange=function() {
	    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
	      document.getElementById("Survey_show").innerHTML=xmlhttp.responseText;
	    }
	  }
	  xmlhttp.send();
}

getSurvey(0);
var question_ans= [];
var question_total = 4;
function Survey_save(id, question_num){
	/* window.alert(document.getElementById(id).value); */
	question_ans[question_num] = document.getElementById(id).value;
}
function Survey_save_radio(id, question_num){
	var obj = document.getElementsByName(id);
	for(var i = 0;i<obj.length;i++){
		if(obj[i].checked == true)
			question_ans[question_num] = obj[i].value;
	}
	/* window.alert(question_ans[question_num-1]); */
}


function Show(){
	url = "/TeamProject/CustomInfo";
	jQuery.ajaxSettings.traditional = true;

	$.ajax({
	    method      : 'POST',
	    url         : url,
	    data        : {
	        'QA' : question_ans
	    },
/* 	    error       : function(request, status, error) {
	    },
	    success     : function(data) {
	    	alert("success");
	    } */
	});
}
function regionUpdate(answer){
	question_ans[0]=answer;
	Show();
}
function MoveUrl(){
	url ="/TeamProject/Recommend?region="+question_ans[0]+"구&age="+question_ans[1]+"&context="+question_ans[2]+"&action=selectfive";

	window.location.href=url;
}
<% %>
</script>
<div id="Survey_show"></div>
<div id = "sample_show"></div>
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