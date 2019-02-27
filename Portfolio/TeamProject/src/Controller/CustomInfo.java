package Controller;

import java.io.IOException;
import java.util.ArrayList;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import Model.dao.CustomInfoDAO;
import Model.vo.CustomInfoVO;
import Model.vo.RegionDataVO;

@WebServlet("/CustomInfo")
public class CustomInfo extends HttpServlet {


	
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String action = request.getParameter("action");
		HttpSession session = request.getSession(true);
		CustomInfoDAO dao = new CustomInfoDAO();
		if(action!=null&&action.equals("logout")) {
			session.setAttribute("login-status", "logout");
			session.setAttribute("id", "logout");
			session.setAttribute("password", "logout");
			
			response.sendRedirect("/TeamProject/jsp/Main.jsp");
			return;
		}
		if(action!=null&&action.equals("read")&&session.getAttribute("id")!=null&&((String)session.getAttribute("login-status")).equals("login"))
		{
			System.out.println("Read!!");
			ArrayList<String> list=dao.read((String)session.getAttribute("id"));
			request.setAttribute("list", list);
			request.getRequestDispatcher("/Recommend?action=myplace").forward(request, response);
			return;
			
		}
		if(action!=null&&action.equals("listone")&&session.getAttribute("id")!=null&&((String)session.getAttribute("login-status")).equals("login"))
		{
			System.out.println("Read!!");
			CustomInfoVO vo=dao.listone((String)session.getAttribute("id"));
			request.setAttribute("listone", vo);
			request.getRequestDispatcher("/jsp/Recommend.jsp").forward(request, response);
			return;
		}
		
		if(action!=null&&action.equals("survey-result")&&session.getAttribute("id")!=null&&((String)session.getAttribute("login-status")).equals("login"))
		{
			System.out.println("survey-result!!");
			CustomInfoVO vo=dao.listone((String)session.getAttribute("id"));
			session.setAttribute("survey-result", vo);
			request.getRequestDispatcher("/jsp/SearchPlace.jsp").forward(request, response);
			return;
		}
		
		if(action!=null&&action.equals("regionread")) {
			float[][] vo= dao.regionread();
			session.setAttribute("regiondata", vo);
			response.sendRedirect("/TeamProject/jsp/Main.jsp");
			return;
		}
		if(action!=null&&action.equals("plusmyplace")&&
				request.getParameter("url")!=null&&
				request.getParameter("myplaceid")!=null&&
				session.getAttribute("id")!=null) {
			System.out.println("plusmyplace");
			String urlmain = request.getParameter("url");
			String myplaceid = request.getParameter("myplaceid");
			String url = "/TeamProject/jsp/"+urlmain+".jsp";
			
			CustomInfoVO vo = dao.listone((String)(session.getAttribute("id")));
			vo.setMyplaceid(myplaceid);
			dao.myplaceinsert(vo);
			
			response.sendRedirect(url);
			return;
		}

		response.sendRedirect("/TeamProject/jsp/Main.jsp");
		return;
	}


	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		request.setCharacterEncoding("UTF-8");
		String action = request.getParameter("action");
		HttpSession session = request.getSession(true);
		String[] list = null;
		CustomInfoDAO dao = new CustomInfoDAO();
		CustomInfoVO vo = new CustomInfoVO();
		String[] question_ans;
		
		if(action!=null&&action.equals("Join"))
		{
			vo.setId(request.getParameter("id"));
			vo.setPassword(request.getParameter("password"));
			vo.setName(request.getParameter("name"));
			System.out.println("join:" +vo.getId()+" "+vo.getName()+" "+vo.getPassword());
			if(dao.IDcheck(request.getParameter("id"))) {
				session.setAttribute("Join-status", "Join-fail");
				response.sendRedirect("/TeamProject/jsp/Join.jsp");
				return;
			}
			dao.Join(vo);
			session.setAttribute("Join-status", "Join-success");
			response.sendRedirect("/TeamProject/jsp/Main.jsp");
			return;
		}
		
		if(action!=null&&action.equals("login")) {
			String id = request.getParameter("id");
			String password = request.getParameter("password");
			System.out.println(id);
			System.out.println(password);
			if(dao.Logincheck(id,password)) {
				System.out.println(session.getAttribute("id"));
				session.setAttribute("id", request.getParameter("id"));
				session.setAttribute("password", request.getParameter("password"));
				session.setAttribute("login-status", "login");
			}else {
				System.out.println(session.getAttribute("id"));
				session.setAttribute("login-status", "fail");
			}
			response.sendRedirect("/TeamProject/jsp/Main.jsp");
			return;
		}
		if(request.getParameter("QA")!=null) {
			
			System.out.println("surveyupdate �떎�뻾!");
			question_ans = request.getParameterValues("QA");
			if(question_ans!=null) {
			for(String str : question_ans)
				System.out.println(str);
			System.out.println((String)session.getAttribute("id"));
			System.out.println(question_ans==null);
			vo.setId((String)session.getAttribute("id"));
			System.out.println((String)session.getAttribute("id"));
			vo.setQ1(question_ans[0]);
			System.out.println(question_ans[0]);
			vo.setQ2(question_ans[1]);
			vo.setQ3(question_ans[2]);
			dao.SurveyUpdate(vo);

			}else {
				System.out.println("survey update = NULL!");
				response.sendRedirect("/TeamProject/jsp/SurveyContent.jsp");
				return;
			}
			session.setAttribute("survey-result", vo);
			response.sendRedirect("/TeamProject/jsp/SurveyContent.jsp");
			return;
		}

		
		/* list = dao.MyplaceIdList(id, password); */
		request.setAttribute("list", list);
	
		request.getRequestDispatcher("/jsp/review.jsp").forward(request, response);
		
	}

}
