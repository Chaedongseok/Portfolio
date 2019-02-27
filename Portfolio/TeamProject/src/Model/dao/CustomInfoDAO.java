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

public class CustomInfoDAO {
	@SuppressWarnings("null")
	public String[] MyplaceIdList(String id) {// 珥덇린 �옣諛붽뎄�땲 �솗�씤
		String[] MyplaceId_list = null;
		int cnt = 0;
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select myplaceid from CustomInfo where id =?");
			pstmt.setString(1, id);
			ResultSet rs = pstmt.executeQuery();
			while (rs.next()) {
				MyplaceId_list[cnt] = rs.getString("myplaceid");
			}
			conn.close();
			rs.close();
			pstmt.close();
		} catch (Exception e) {
			System.out.println("由ъ뒪�듃�뿉�윭");
			e.printStackTrace();

		}
		return MyplaceId_list;
	}

	public CustomInfoVO listone(String id) {
		CustomInfoVO vo = new CustomInfoVO();
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("Select * from CustomInfo where id =?");
			pstmt.setString(1, id);
			pstmt.executeUpdate();
			ResultSet rs = pstmt.executeQuery();
			conn.commit();
			if(rs.next()) {
				vo.setId(rs.getString("id"));
				vo.setPassword(rs.getString("password"));
				vo.setName(rs.getString("name"));
				vo.setQ1(rs.getString("q1"));
				vo.setQ2(rs.getString("q2"));
				vo.setQ3(rs.getString("q3"));
				vo.setMyplaceid(rs.getString("myplaceid"));
			}
			pstmt.close();
			conn.close();
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
			System.out.println("LISTONE �삤瑜�!");
		} catch (NamingException ne) {
			ne.printStackTrace();
			System.out.println("listone �꽕�씠諛띿삤瑜�");

		}
		return vo;
	}

	public ArrayList<String> read(String id) {
		ArrayList<String> str = new ArrayList<String>();
	
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("Select myplaceid from CustomInfo where id =?");
			System.out.println("id : " + id);
			pstmt.setString(1, id);
			pstmt.executeUpdate();
			ResultSet rs = pstmt.executeQuery();
			conn.commit();
			while(rs.next()) {
				str.add(rs.getString("myplaceid"));
				System.out.println(rs.getString("myplaceid"));
			}
			pstmt.close();
			conn.close();
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
			System.out.println("LISTONE 오류!");
		} catch (NamingException ne) {
			ne.printStackTrace();
			System.out.println("listone namingexception error");
		}
		return str;
	}
	
	public float[][] regionread(){
		float[][] arr = new float[21][24];
		try {
			Connection conn = connectDB();
			Statement stmt = conn.createStatement();
			ResultSet rs=stmt.executeQuery("select * from regionpred"); 
			int j=0;
			while(rs.next()) {
				for(int i = 0;i<24;i++)
					arr[j][i]=rs.getFloat("T"+(i+1));
				j++;
			}
			closeDB(conn,stmt,rs);
		} catch (SQLException e) {
			e.printStackTrace();
			System.out.println("LISTONE �삤瑜�!");
		} catch (NamingException ne) {
			ne.printStackTrace();
			System.out.println("listone �꽕�씠諛띿삤瑜�");
		}
		return arr;
	}


	public boolean SurveyUpdate(CustomInfoVO vo) { 
		try { 
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement(
					"Update CustomInfo set q1=?, q2=?,q3=? where id =?");
			pstmt.setString(1, vo.getQ1());
			pstmt.setString(2, vo.getQ2());
			pstmt.setString(3, vo.getQ3());
			pstmt.setString(4, vo.getId());
			pstmt.executeUpdate();
			conn.commit();
			System.out.println("SurveyUpdate실행");
			pstmt.close();
			conn.close();
		}catch(Exception sqle) { 
			sqle.printStackTrace();
			System.out.println("SurveyUpdate오류"); 
			return false; 
		} 
		return true; 
	}
	public boolean myplaceinsert(CustomInfoVO vo) {
		try { 
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement(
					"Insert into CustomInfo values(?,?,?,?,?,?,?,?)");
			pstmt.setString(1, vo.getId());
			pstmt.setString(2, vo.getName());
			pstmt.setString(3, vo.getPassword());
			pstmt.setString(4, vo.getQ1());
			pstmt.setString(5, vo.getQ2());
			pstmt.setString(6, vo.getQ3());
			pstmt.setString(7, vo.getQ4());
			pstmt.setString(8, vo.getMyplaceid());
			pstmt.executeUpdate();
			conn.commit();
			System.out.println("myplaceinsert실행");
			pstmt.close();
			conn.close();
		}catch(Exception sqle) { 
			sqle.printStackTrace();
			System.out.println("myplaceinsert오류"); 
			return false; 
		} 
		return true; 
	}

	public boolean Join(CustomInfoVO vo) {// �옣諛붽뎄�땲 異붽�
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("Insert into CustomInfo values(?,?,?,0,0,0,0,0)");

			pstmt.setString(1, vo.getId());
			pstmt.setString(2, vo.getPassword());
			pstmt.setString(3, vo.getName());
			pstmt.executeUpdate();

			conn.commit();
			pstmt.close();
			conn.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Join �삤瑜� !");
			return false;
		}
		return true;
	}

	public boolean delete(String id, String myplaceid) {// �옣諛붽뎄�땲 �궘�젣
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("DELETE FROM CustomInfo WHERE id = ? AND myplaceid=?");
			pstmt.setString(1, id);
			pstmt.setString(2, myplaceid);
			pstmt.executeUpdate();

			conn.commit();
			pstmt.close();
			conn.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("delete �삤瑜� !");
			return false;
		}
		return true;
	}

	public boolean Logincheck(String id, String password) {// �옣諛붽뎄�땲 �궘�젣
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select * from customInfo where id=? and password=?");
			pstmt.setString(1, id);
			pstmt.setString(2, password);
			pstmt.executeUpdate();
			ResultSet rs = pstmt.executeQuery();

			if (rs.next()) {
				System.out.println("ID 있음");
				rs.close();
				conn.commit();
				pstmt.close();
				conn.close();
				return true;
			}
			rs.close();
			conn.commit();
			pstmt.close();
			conn.close();
			System.out.println("회원가입이 안됨");
			return false;

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("LoginCheck �삤瑜� !");
			return false;
		}
	}

	public boolean IDcheck(String id) {// �옣諛붽뎄�땲 �궘�젣
		try {
			Connection conn = connectDB();
			PreparedStatement pstmt = conn.prepareStatement("select * from customInfo where id=?");
			pstmt.setString(1, id);
			pstmt.executeUpdate();
			ResultSet rs = pstmt.executeQuery();

			if (rs.next()) {
				System.out.println("ID �엳�뵲");
				rs.close();
				conn.commit();
				pstmt.close();
				conn.close();
				return true;
			}
			rs.close();
			conn.commit();
			pstmt.close();
			conn.close();
			System.out.println("Id �뾾�떎");
			return false;
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("IDcheck �삤瑜� !");
			return false;
		}
	}

	public Connection connectDB() throws NamingException {
		Connection conn = null;
		try {
			Class.forName("oracle.jdbc.OracleDriver");
			conn = DriverManager.getConnection("jdbc:oracle:thin:@70.12.113.169:1521:xe", "java", "java");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("connectDB error");
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
			System.out.println("closeDB error");
			return false;
		}
		return true;
	}
}
