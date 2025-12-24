"""
Traffic Sign Mapping - Vietnamese Translation and Classification
Based on: QCVN 41:2019 & MobileNetV2 Dataset (57 Classes)
"""

# 1. TỪ ĐIỂN Ý NGHĨA (MÃ -> TIẾNG VIỆT)
SIGN_CODE_TO_NAME = {
    # --- Cấm ---
    "DP.135": "Hết tất cả các lệnh cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm xe ô tô",
    "P.103b": "Cấm xe ô tô rẽ phải",
    "P.103c": "Cấm xe ô tô rẽ trái",
    "P.104": "Cấm xe máy",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2.5T",
    "P.107a": "Cấm xe khách",
    "P.111": "Cấm xe ba gác",
    "P.112": "Cấm người đi bộ",
    "P.115": "Hạn chế trọng tải xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm quay đầu xe",
    "P.124b": "Cấm ô tô quay đầu",
    "P.124c": "Cấm rẽ trái và quay đầu",
    "P.125": "Cấm vượt",
    "P.127": "Tốc độ tối đa cho phép",
    "P.128": "Cấm sử dụng còi",
    "P.129": "Kiểm tra",
    "P.130": "Cấm dừng và đỗ xe",
    "P.131a": "Cấm đỗ xe",
    "P.137": "Cấm rẽ trái và rẽ phải",
    "P.245a": "Đi chậm (Biển P)",

    # --- Hiệu Lệnh (R) ---
    "R.301c": "Các xe chỉ được rẽ trái",
    "R.301d": "Các xe chỉ được rẽ phải",
    "R.301e": "Các xe chỉ được rẽ trái (biển phụ)",
    "R.302a": "Vòng chướng ngại vật sang phải",
    "R.302b": "Vòng chướng ngại vật sang trái",
    "R.303": "Nơi giao nhau chạy theo vòng xuyến",
    "R.407a": "Đường một chiều",
    "R.409": "Chỗ quay xe",
    "R.425": "Cầu vượt đi bộ",
    "R.434": "Bến xe buýt",

    # --- Nguy Hiểm (W) & Chỉ Dẫn (S) ---
    "S.509a": "Thuyết minh biển chính",
    "W.201a": "Ngoặt trái nguy hiểm",
    "W.201b": "Ngoặt phải nguy hiểm",
    "W.202a": "Nhiều chỗ ngoặt liên tiếp",
    "W.202b": "Nhiều chỗ ngoặt (phải)",
    "W.203a": "Đường bị thu hẹp", # (Lưu ý: dataset có thể dùng W.203a hoặc b, c)
    "W.203b": "Đường hẹp bên trái",
    "W.203c": "Đường hẹp bên phải",
    "W.205a": "Đường giao nhau cùng cấp",
    "W.205b": "Giao nhau cùng cấp (phải)",
    "W.205d": "Giao nhau cùng cấp (trái)",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207b": "Giao đường không ưu tiên (phải)",
    "W.207c": "Giao đường không ưu tiên (trái)",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có đèn tín hiệu",
    "W.210": "Giao nhau với đường sắt",
    "W.219": "Dốc xuống nguy hiểm",
    "W.221b": "Đường có gồ giảm tốc",
    "W.224": "Người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.227": "Công trường",
    "W.233": "Nguy hiểm khác",
    "W.235": "Đường đôi",
    "W.245a": "Đi chậm (Biển W)",
    "W.246a": "Chú ý chướng ngại vật - Vòng tránh sang hai bên" # (Check log nếu có folder này)
}

# 2. DANH SÁCH 57 LỚP (Thứ tự ABC từ thư mục train)
# QUAN TRỌNG: Danh sách này phải khớp 100% với folder train
SORTED_CLASS_FOLDERS = [
    "DP.135",
    "P.102", "P.103a", "P.103b", "P.103c", "P.104", 
    "P.106a", "P.106b", "P.107a", 
    "P.111", "P.112", "P.115", "P.117", 
    "P.123a", "P.123b", 
    "P.124a", "P.124b", "P.124c", "P.125", "P.127", "P.128", "P.129", 
    "P.130", "P.131a", "P.137", 
    "P.245a", 
    "R.301c", "R.301d", "R.301e", 
    "R.302a", "R.302b", "R.303", 
    "R.407a", "R.409", "R.425", "R.434", 
    "S.509a", 
    "W.201a", "W.201b", 
    "W.202a", "W.202b", 
    "W.203b", "W.203c", 
    "W.205a", "W.205b", "W.205d", 
    "W.207a", "W.207b", "W.207c", 
    "W.208", "W.209", "W.210", "W.219", 
    "W.221b", "W.224", "W.225", "W.227", 
    "W.233", "W.235", "W.245a"
]

def get_cnn_class_name(class_id: int) -> str:
    """
    Chuyển đổi ID lớp (0->56) thành Tên hiển thị
    """
    if 0 <= class_id < len(SORTED_CLASS_FOLDERS):
        code = SORTED_CLASS_FOLDERS[class_id]
        name = SIGN_CODE_TO_NAME.get(code, "Không xác định")
        return f"{code} - {name}"
    return f"Unknown_{class_id}"

def translate_sign_name(class_name_en: str) -> str:
    """Hàm dịch cho YOLO"""
    return SIGN_CODE_TO_NAME.get(class_name_en, class_name_en)