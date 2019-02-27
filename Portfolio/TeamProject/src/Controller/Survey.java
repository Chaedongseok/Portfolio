package Controller;

import java.io.IOException;
import java.util.ArrayList;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class Survey
 */
@WebServlet("/Survey")
public class Survey extends HttpServlet {
	private static final long serialVersionUID = 1L;
	String[] question_ans;
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		System.out.println("GET방식 호출");
		doPost(request,response);
	}
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		System.out.println("PSOT호출");
		question_ans = request.getParameterValues("QA");
		/*
		 * DB에 저장하는 기능이 필요함!
		 * 
		 * 
		 * */
		System.out.println(question_ans==null);
		request.setAttribute("question_ans",question_ans);
		request.setAttribute("Survey_status","Result");
		request.setAttribute("question_num", null);
		request.getRequestDispatcher("/jsp/SurveyContent.jsp").forward(request, response);
		System.out.println("Post 전송 끝");
	}

}
