package Model.vo;

public class SearchPlaceVO {
	private String newaddress;
	private String oldaddress;
	private String x;
	private String y;
	private String content_code;
	private String content_name;
	public String getContent_name() {
		return content_name;
	}
	public void setContent_name(String content_name) {
		this.content_name = content_name;
	}
	private String region;
	private String phonenum;
	private String category;
	public String getNewaddress() {
		return newaddress;
	}
	public void setNewaddress(String newaddress) {
		this.newaddress = newaddress;
	}
	public String getOldaddress() {
		return oldaddress;
	}
	public void setOldaddress(String oldaddress) {
		this.oldaddress = oldaddress;
	}
	public String getX() {
		return x;
	}
	public void setX(String x) {
		this.x = x;
	}
	public String getY() {
		return y;
	}
	public void setY(String y) {
		this.y = y;
	}
	public String getContent_code() {
		return content_code;
	}
	public void setContent_code(String content_code) {
		this.content_code = content_code;
	}
	public String getRegion() {
		return region;
	}
	public void setRegion(String region) {
		this.region = region;
	}
	public String getPhonenum() {
		return phonenum;
	}
	public void setPhonenum(String phonenum) {
		this.phonenum = phonenum;
	}
	public String getCategory() {
		return category;
	}
	public void setCategory(String category) {
		this.category = category;
	}
	
}
