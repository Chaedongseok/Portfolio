package Model.dao;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;

import javax.naming.NamingException;

import Model.vo.CustomInfoVO;
import Model.vo.RecommendVO;
import Model.vo.SearchPlaceVO;

public class RecommendDAO {
	
	public ArrayList<RecommendVO> selectfive(String region,String age, String context) {// 珥덇린 �옣諛붽뎄�땲 �솗�씤
		ArrayList<RecommendVO> list = new ArrayList<RecommendVO>();
		try {
			System.out.println("selectivefive �떎�뻾");
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select * from "+region+" where context =? and age=? ");
			//�뿉�윭�굹硫� �솗�씤!--------------------------------------------
			pstmt.setString(1, context);
			pstmt.setString(2, age);
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				RecommendVO vo = new RecommendVO();
				vo.setPlace(rs.getString("place"));
				vo.setAge(rs.getString("age"));
				vo.setContext(rs.getString("context"));
				vo.setPrice(rs.getString("price"));
				vo.setCate_f(rs.getString("cate_f"));
				vo.setReview(rs.getString("review"));
				vo.setPicture1(rs.getString("picture1"));
				vo.setMenu(rs.getString("menu"));
				vo.setX(rs.getString("x"));
				vo.setY(rs.getString("y"));
				vo.setFeeling(rs.getString("feeling"));
				vo.setHomepage(rs.getString("homepage"));
				vo.setWorktime(rs.getString("worktime"));
				list.add(vo);
				System.out.println(vo.getAge());
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectFIve ERROR!----");
			e.printStackTrace();
		}
		return list;
	}
	
	public ArrayList<SearchPlaceVO> selectread(String region, String category){
		ArrayList<SearchPlaceVO> list = new ArrayList<SearchPlaceVO>();
		try {
			System.out.println(region);
			System.out.println(category);
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_코드,콘텐츠_명,region,전화번호,category from MUNHWA where region =? and category=?");
			//�뿉�윭�굹硫� �솗�씤!--------------------------------------------
			pstmt.setString(1, region);
			pstmt.setString(2, category);
			
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				SearchPlaceVO vo = new SearchPlaceVO();
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
				list.add(vo);
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectread ERROR!----");
			e.printStackTrace();
		}
		
		return list;
	}
	public SearchPlaceVO selectreadbymyplaceid(String myplaceid){
		SearchPlaceVO vo = new SearchPlaceVO();
		try {
			
			System.out.println(myplaceid);
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_코드,콘텐츠_명,region,전화번호,category from MUNHWA where 콘텐츠_코드=?");
			//�뿉�윭�굹硫� �솗�씤!--------------------------------------------
			pstmt.setString(1, myplaceid);
			
			ResultSet rs = pstmt.executeQuery();
			if (rs.next()) {
				System.out.println("selectreadbymyplaceid start!--------------");
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectreadbymyplaceid ERROR!----");
			e.printStackTrace();
		}
		
		return vo;
	}
	public ArrayList<SearchPlaceVO> selectreadall(){
		ArrayList<SearchPlaceVO> list = new ArrayList<SearchPlaceVO>();
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_명,콘텐츠_코드,region,전화번호,category from MUNHWA");
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				SearchPlaceVO vo = new SearchPlaceVO();
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
				list.add(vo);
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectread ERROR!----");
			e.printStackTrace();
		}
		
		return list;
	}
	public ArrayList<SearchPlaceVO> selectreadallcat(String region){
		ArrayList<SearchPlaceVO> list = new ArrayList<SearchPlaceVO>();
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_명,콘텐츠_코드,region,전화번호,category from MUNHWA where region =?");
			pstmt.setString(1, region);
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				SearchPlaceVO vo = new SearchPlaceVO();
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
				list.add(vo);
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectread ERROR!----");
			e.printStackTrace();
		}
		
		return list;
	}
	
	public ArrayList<SearchPlaceVO> selectreadallregion(String category){
		ArrayList<SearchPlaceVO> list = new ArrayList<SearchPlaceVO>();
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_명,콘텐츠_코드,region,전화번호,category from MUNHWA where category =?");
			pstmt.setString(1, category);
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				SearchPlaceVO vo = new SearchPlaceVO();
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
				list.add(vo);
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectread ERROR!----");
			e.printStackTrace();
		}
		
		return list;
	}
	public SearchPlaceVO selectone(String category_code) {
		SearchPlaceVO vo = new SearchPlaceVO();
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select 새주소,지번주소,좌표정보,좌표정보2,콘텐츠_명,콘텐츠_코드,region,전화번호,category from MUNHWA where 콘텐츠_코드 =?");
			pstmt.setString(1, category_code);
			ResultSet rs = pstmt.executeQuery();
			if (rs.next()) {
				vo.setNewaddress(rs.getString("새주소"));
				vo.setOldaddress(rs.getString("지번주소"));
				vo.setX(rs.getString("좌표정보"));
				vo.setY(rs.getString("좌표정보2"));
				vo.setContent_name(rs.getString("콘텐츠_명"));
				vo.setContent_code(rs.getString("콘텐츠_코드"));
				vo.setRegion(rs.getString("region"));
				vo.setPhonenum(rs.getString("전화번호"));
				vo.setCategory(rs.getString("category"));
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("selectread ERROR!----");
			e.printStackTrace();
		}
		
		return vo;
		
	}
	public Connection connectDB() throws NamingException {
		Connection conn = null;
		try {
			Class.forName("oracle.jdbc.OracleDriver");
			conn = DriverManager.getConnection("jdbc:oracle:thin:@70.12.113.169:1521:xe", "resinfo", "resinfo");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("�뱶�씪�씠鍮� �삤瑜�");
		}
		return conn;
	}

	public boolean closeDB(Connection conn, Statement stmt, ResultSet rs) {
		try {
			conn.close();
			stmt.close();
			rs.close();
		} catch (SQLException sqle) {
			sqle.printStackTrace();
			System.out.println("closeDB�삤瑜�");
			return false;
		}
		return true;
	}
}
