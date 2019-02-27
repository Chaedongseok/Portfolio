package Controller;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import Model.dao.RecommendDAO;
import Model.vo.RecommendVO;
import Model.vo.SearchPlaceVO;


@WebServlet("/Recommend")
public class Recommend extends HttpServlet {
	private static final long serialVersionUID = 1L;
	ArrayList<RecommendVO> list = new ArrayList<RecommendVO>();
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		HttpSession session = request.getSession(true);
		RecommendDAO dao = new RecommendDAO();
		if(request.getParameter("region")!=null&&
			request.getParameter("age")!=null&&
			request.getParameter("context")!=null&&
			request.getParameter("action")!=null&&
			request.getParameter("action").equals("selectfive")) {
			System.out.println("Recommend servlet listall 실행");
			HashMap<String, String> contextmap = new HashMap<String, String>();
			contextmap.put("혼자","13");
			contextmap.put("연인","1");
			contextmap.put("친구","14");
			contextmap.put("가족","15");
			contextmap.put("모임","2");
			HashMap<String, String> regionmap = new HashMap<String, String>();
			regionmap.put("종로구", "GANGDONG");
			regionmap.put("중랑구", "GANGDONG");
			regionmap.put("서대문구", "GANGDONG");
			regionmap.put("영등포구", "GANGDONG");
			regionmap.put("중구", "GANGDONG");
			regionmap.put("성북구", "GANGDONG");
			regionmap.put("마포구", "GANGDONG");
			regionmap.put("동작구", "GANGDONG");
			regionmap.put("용산구", "GANGDONG");
			regionmap.put("강북구", "GANGBUK");
			regionmap.put("양천구", "GANGDONG");
			regionmap.put("관악구", "GWANAK");
			regionmap.put("광진구", "GWANGJIN");
			regionmap.put("노원구", "NOWON");
			regionmap.put("구로구", "GURO");
			regionmap.put("강남구", "GANGDONG");
			regionmap.put("동대문구", "DONGDAEMUN");
			regionmap.put("은평구", "EUNPYEONG");
			regionmap.put("금천구", "GEUMCHEON");
			regionmap.put("송파구", "SONGPA");
			regionmap.put("강동구", "GANGDONG");
			
			
			
			System.out.println(request.getParameter("context"));
			String region = regionmap.get(request.getParameter("region"));
			String age = request.getParameter("age");
			String context = contextmap.get(request.getParameter("context"));
			System.out.println("region :" +region);
			System.out.println("age :" +age);
			System.out.println("context :" +context);
			list = dao.selectfive(region, age, context);
			session.setAttribute("selectfive", list);
			response.sendRedirect("/TeamProject/jsp/Recommend.jsp");
			return;
		}
		if(request.getParameter("region")!=null&&
			request.getParameter("category")!=null&&
			request.getParameter("action")!=null&&
			request.getParameter("action").equals("searchplace")) {
			
			String region = request.getParameter("region");
			String category = request.getParameter("category");
			
			System.out.println("region : " + region);
			System.out.println("category : " + category);
			ArrayList<SearchPlaceVO> list = dao.selectread(region, category);
			session.setAttribute("searchplace", list);
			response.sendRedirect("/TeamProject/jsp/PlaceContentAjax.jsp");
			return;
		}
		if(		request.getParameter("action")!=null&&
				request.getParameter("action").equals("searchplaceall")) {
				ArrayList<SearchPlaceVO> list = dao.selectreadall();
				session.setAttribute("searchplace", list);
				response.sendRedirect("/TeamProject/jsp/PlaceContentAjax.jsp");
				return;
			}
		if(		request.getParameter("region")!=null&&
				request.getParameter("action")!=null&&
				request.getParameter("action").equals("searchplacecat")) {
			
				ArrayList<SearchPlaceVO> list = dao.selectreadallcat(request.getParameter("region"));
				session.setAttribute("searchplace", list);
				response.sendRedirect("/TeamProject/jsp/PlaceContentAjax.jsp");
				return;
			}
		
		if(		request.getParameter("category")!=null&&
				request.getParameter("action")!=null&&
				request.getParameter("action").equals("searchplaceregion")) {
				System.out.println("카데고리별 실행!");
				System.out.println("");
				ArrayList<SearchPlaceVO> list = dao.selectreadallregion(request.getParameter("category"));
				session.setAttribute("searchplace", list);
				response.sendRedirect("/TeamProject/jsp/PlaceContentAjax.jsp");
				return;
			}
		
		if(request.getParameter("content-code")!=null&&
				request.getParameter("action")!=null&&
				request.getParameter("action").equals("showdetail")) {
			SearchPlaceVO vo = new SearchPlaceVO();
			vo = dao.selectone(request.getParameter("content-code"));
			System.out.println("selectone 실행");
			request.setAttribute("selectone", vo);
			request.getRequestDispatcher("/jsp/PlaceContent.jsp").forward(request, response);
			return;
		}
		
		if(request.getParameter("action")!=null&&
				request.getParameter("action").equals("myplace")&&
				request.getAttribute("list")!=null) {
			ArrayList<String> list =(ArrayList<String>)request.getAttribute("list");
			ArrayList<SearchPlaceVO> listSP = new ArrayList<SearchPlaceVO>();
			for(String myplaceid : list) {
				SearchPlaceVO vo = new SearchPlaceVO();
				vo=dao.selectreadbymyplaceid(myplaceid);
				listSP.add(vo);
			}
			request.setAttribute("list",listSP);
			request.getRequestDispatcher("/jsp/Myplace.jsp").forward(request, response);
			return;
				
		}
		System.out.println("doget 아무일도 없었음");
		response.sendRedirect("/TeamProject/jsp/Main.jsp");
	}
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
	
	
	}

}
