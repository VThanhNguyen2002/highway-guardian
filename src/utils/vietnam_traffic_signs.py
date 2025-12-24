"""
Vietnam Traffic Signs Mapping - QCVN 41:2019/BGTVT
Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ
Integrated with YOLO and MobileNetV2 CNN mappings
"""

# ============================================================================
# MASTER CLASS NAMES - Final merged classes from all datasets
# ============================================================================
MASTER_CLASS_NAMES = [
    'DP.135', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106b',
    'P.107a', 'P.111', 'P.112', 'P.115', 'P.117', 'P.123a', 'P.123b', 'P.124a',
    'P.124b', 'P.124c', 'P.124d', 'P.125', 'P.126', 'P.127', 'P.128', 'P.129',
    'P.130', 'P.131a', 'P.137', 'P.139', 'P.245a', 'R.122', 'R.301a', 'R.301c',
    'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303', 'R.306', 'R.407a', 'R.409',
    'R.415a', 'R.425', 'R.434', 'S.505a', 'S.509a', 'W.201a', 'W.201b', 'W.202a',
    'W.202b', 'W.203a', 'W.203b', 'W.203c', 'W.205a', 'W.205b', 'W.205c', 'W.205d',
    'W.207', 'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210', 'W.219',
    'W.221b', 'W.222a', 'W.224', 'W.225', 'W.227', 'W.233', 'W.235', 'W.239b',
    'W.245a', 'W.246a', 'W.246c'
]

# ============================================================================
# CNN (MobileNetV2) CLASS ID MAPPING
# Based on training data - maps class index to sign code
# ============================================================================
CNN_CLASS_ID_TO_CODE = {
    # This will be populated based on your CNN training order
    # Example structure (you need to provide the actual mapping):
    # 0: 'P.102',
    # 1: 'P.103a',
    # ... etc
}

# ============================================================================
# YOLO CLASS NAME MAPPING
# Maps YOLO class names to standardized sign codes
# ============================================================================
YOLO_CLASS_TO_CODE = {
    # From vietnam-traffic-signs-kaggle
    'W.224': 'W.224', 'W.205c': 'W.205c', 'P.102': 'P.102', 'R.302a': 'R.302a',
    'W.205a': 'W.205a', 'W.207': 'W.207', 'W.201a': 'W.201a', 'P.123a': 'P.123a',
    'I.434a': 'R.434', 'R.303': 'R.303', 'P.130': 'P.130', 'I.409': 'R.409',
    'R.415a': 'R.415a', 'W.245a': 'W.245a', 'P.106a*Xe tải': 'P.106a',
    'W.203c': 'W.203c', 'P.117*': 'P.117', 'P.124a*': 'P.124a', 'P.107': 'P.107a',
    'P.124d': 'P.124d', 'P.103a': 'P.103a', 'W.203b': 'W.203b', 'W.221b': 'W.221b',
    'P.111': 'P.111', 'P.129': 'P.129', 'S.505a*Xe máy': 'S.505a', 'W.246a': 'W.246a',
    'W.225': 'W.225', 'S.505a*Xe tải và công': 'S.505a', 'P.104': 'P.104',
    'S.505a*Xe tải': 'S.505a', 'P.123b': 'P.123b', 'W.202b': 'W.202b',
    'P.137': 'P.137', 'P.139': 'P.139', 'W.205b': 'W.205b', 'R.301e': 'R.301e',
    'W.239b*': 'W.239b', 'W.233': 'W.233', 'I.407a': 'R.407a', 'P.131a': 'P.131a',
    'P.124b1': 'P.124b', 'W.210': 'W.210', 'P.124c': 'P.124c', 'W.201b': 'W.201b',
    'W.246c': 'W.246c',
    # Speed limits - all map to P.127
    'P.127*50': 'P.127', 'P.127*60': 'P.127', 'P.127*80': 'P.127', 'P.127*40': 'P.127',
    
    # From traffic-and-road-signs (English)
    '-Road narrows on right': 'W.203c', '50 mph speed limit': 'P.127',
    'Beware of children': 'W.225', 'Dangerous Left Curve Ahead': 'W.201a',
    'Dangerous Rright Curve Ahead': 'W.201b', 'End of all speed and passing limits': 'DP.135',
    'Give Way': 'W.208', 'Go Straight or Turn Right': 'R.301d', 'Keep-Left': 'R.302b',
    'Keep-Right': 'R.302a', 'Left Zig Zag Traffic': 'W.202b', 'No Entry': 'P.102',
    'No_Over_Taking': 'P.125', 'Overtaking by trucks is prohibited': 'P.126',
    'Pedestrian Crossing': 'W.224', 'Round-About': 'R.303', 'Slippery Road Ahead': 'W.222a',
    'Speed Limit 20 KMPh': 'P.127', 'Speed Limit 30 KMPh': 'P.127', 'Stop_Sign': 'R.122',
    'Straight Ahead Only': 'R.301a', 'Traffic_signal': 'W.209',
    'Truck traffic is prohibited': 'P.106a', 'Turn left ahead': 'R.301c',
    'Uneven Road': 'W.221b',
    
    # From vietnam-traffic-sign-8k8 (Vietnamese without accents)
    'Bien bao ben xe bus': 'R.434', 'Bien bao cam di nguoc chieu': 'P.102',
    'Bien bao cam do xe': 'P.131a', 'Bien bao cam dung xe va do xe': 'P.130',
    'Bien bao cam nguoi di bo': 'P.112', 'Bien bao cam quay dau xe': 'P.124a',
    'Bien bao cam re phai': 'P.123b', 'Bien bao cam re trai': 'P.123a',
    'Bien bao cam vuot': 'P.125', 'Bien bao cam xe may': 'P.104',
    'Bien bao cam xe oto': 'P.103a', 'Bien bao cam xe oto tai': 'P.106a',
    'Bien bao cho ngoat nguy hiem': 'W.201a', 'Bien bao cho quay xe': 'R.409',
    'Bien bao cong truong': 'W.227', 'Bien bao di cham': 'W.245a',
    'Bien bao duong bi thu hep': 'W.203a', 'Bien bao duong nguoi di bo cat ngang': 'W.224',
    'Bien bao giao nhau co tin hieu den': 'W.209',
    'Bien bao giao nhau voi duong khong uu tien': 'W.207a',
    'Bien bao giao nhau voi duong uu tien': 'W.208',
    'Bien bao han che chieu cao': 'P.117',
    'Bien bao huong di thang phai theo': 'R.301a',
    'Bien bao noi giao nhau chay theo vong xuyen': 'R.303',
    'Bien bao toc do toi da cho phep': 'P.127',
    'Bien bao toc do toi thieu cho phep': 'R.306',
    'Bien bao tre em': 'W.225',
}

# ============================================================================
# SIGN CODE TO VIETNAMESE NAME MAPPING (QCVN 41:2019)
# ============================================================================
# Biển báo CẤM (P - Prohibitory Signs)
PROHIBITORY_SIGNS = {
    "P.101": "Đường cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm ô tô",
    "P.103b": "Cấm ô tô rẽ phải",
    "P.103c": "Cấm ô tô rẽ trái",
    "P.104": "Cấm mô tô",
    "P.105": "Cấm xe máy",
    "P.106a": "Cấm ô tô và mô tô",
    "P.106b": "Cấm xe máy và xe gắn máy",
    "P.107a": "Cấm xe ô tô tải",
    "P.107b": "Cấm xe ô tô tải có trọng tải trên... tấn",
    "P.108": "Cấm xe ô tô khách và ô tô tải",
    "P.109": "Cấm xe ô tô kéo moóc",
    "P.110": "Cấm xe ô tô kéo rơ moóc",
    "P.111": "Cấm xe kéo",
    "P.112": "Cấm xe thô sơ",
    "P.113": "Cấm người đi bộ",
    "P.114": "Cấm xe đạp",
    "P.115": "Cấm xe súc vật kéo",
    "P.116": "Cấm súc vật đi qua",
    "P.117": "Cấm xe gắn máy",
    "P.118": "Cấm xe lam",
    "P.119": "Cấm xe ba bánh cơ giới",
    "P.120": "Cấm xe ba bánh thô sơ",
    "P.121a": "Cấm rẽ trái",
    "P.121b": "Cấm rẽ phải",
    "P.122": "Cấm quay đầu xe",
    "P.123a": "Cấm ô tô rẽ trái",
    "P.123b": "Cấm ô tô rẽ phải",
    "P.124a": "Cấm rẽ trái và quay đầu xe",
    "P.124b": "Cấm rẽ phải và quay đầu xe",
    "P.125": "Cấm vượt",
    "P.126": "Cấm ô tô tải vượt",
    "P.127": "Cấm xe ô tô khách vượt",
    "P.128": "Cấm còi",
    "P.129": "Cấm dừng xe và đỗ xe",
    "P.130": "Cấm đỗ xe",
    "P.131a": "Cấm đỗ xe vào ngày chẵn",
    "P.131b": "Cấm đỗ xe vào ngày lẻ",
    "P.132": "Cấm xe ô tô khách và ô tô tải",
    "P.133": "Cấm xe ô tô tải",
    "P.134": "Cấm xe ô tô khách",
    "P.135": "Cấm xe ô tô",
}

# Giới hạn tốc độ
SPEED_LIMIT_SIGNS = {
    "P.127": "Tốc độ tối đa cho phép (km/h)",
    "P.128": "Cấm xe có trọng tải trên... tấn",
}

# Biển báo NGUY HIỂM (W - Warning Signs)
WARNING_SIGNS = {
    "W.201": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "W.202": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.203": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.204": "Đường giao nhau",
    "W.205a": "Giao nhau với đường không ưu tiên",
    "W.205b": "Giao nhau với đường ưu tiên",
    "W.205c": "Giao nhau với đường hai chiều",
    "W.205d": "Giao nhau với đường một chiều",
    "W.206": "Giao nhau có tín hiệu đèn",
    "W.207a": "Giao nhau với đường sắt có rào chắn",
    "W.207b": "Giao nhau với đường sắt không có rào chắn",
    "W.207c": "Giao nhau với đường sắt",
    "W.208": "Giao nhau với đường cong",
    "W.209": "Đường người đi bộ cắt ngang",
    "W.210": "Trẻ em",
    "W.211a": "Công trường",
    "W.211b": "Đoạn đường đang thi công",
    "W.212": "Đường hẹp",
    "W.213": "Đường hẹp về phía trái",
    "W.214": "Đường hẹp về phía phải",
    "W.215": "Đường hai chiều",
    "W.216": "Đường có gờ giảm tốc",
    "W.217": "Đường không bằng phẳng",
    "W.218": "Dốc xuống nguy hiểm",
    "W.219": "Dốc lên nguy hiểm",
    "W.220": "Đường trơn",
    "W.221a": "Đá lở",
    "W.221b": "Bờ đá nguy hiểm",
    "W.222a": "Bờ sông, bờ hồ nguy hiểm",
    "W.222b": "Bến phà",
    "W.223": "Cầu hẹp",
    "W.224": "Cầu quay",
    "W.225": "Cầu cất",
    "W.226": "Đường ướt trơn",
    "W.227": "Đường có ổ gà, sống trâu",
    "W.228": "Đường có vật chướng ngại",
    "W.229": "Đường có vật chướng ngại",
    "W.230": "Đường có vật chướng ngại",
    "W.231": "Đường có vật chướng ngại",
    "W.232": "Đường có vật chướng ngại",
    "W.233": "Đường có vật chướng ngại",
    "W.234": "Đường có vật chướng ngại",
    "W.235": "Đường có vật chướng ngại",
    "W.236": "Đường có vật chướng ngại",
    "W.237": "Đường có vật chướng ngại",
    "W.238": "Đường có vật chướng ngại",
    "W.239": "Đường có vật chướng ngại",
    "W.240": "Đường có vật chướng ngại",
    "W.241": "Đường có vật chướng ngại",
    "W.242": "Đường có vật chướng ngại",
    "W.243": "Đường có vật chướng ngại",
    "W.244": "Đường có vật chướng ngại",
    "W.245": "Đường có vật chướng ngại",
    "W.246": "Đường có vật chướng ngại",
    "W.247": "Đường có vật chướng ngại",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sign_code_from_yolo(yolo_class_name: str) -> str:
    """
    Convert YOLO class name to standardized sign code
    
    Args:
        yolo_class_name: Class name from YOLO model
        
    Returns:
        Standardized sign code (e.g., 'P.102') or original name if not found
    """
    return YOLO_CLASS_TO_CODE.get(yolo_class_name, yolo_class_name)

def get_sign_code_from_cnn(class_id: int) -> str:
    """
    Convert CNN class ID to standardized sign code
    
    Args:
        class_id: Class index from CNN model
        
    Returns:
        Standardized sign code (e.g., 'P.102') or 'Unknown'
    """
    return CNN_CLASS_ID_TO_CODE.get(class_id, f'Unknown_{class_id}')

def get_sign_name(sign_code: str) -> str:
    """
    Get Vietnamese name for a sign code
    
    Args:
        sign_code: Sign code (e.g., 'P.102')
        
    Returns:
        Vietnamese name of the sign
    """
    # Search in all sign dictionaries
    all_signs = {
        **PROHIBITORY_SIGNS,
        **WARNING_SIGNS,
        **MANDATORY_SIGNS,
        **INFORMATION_SIGNS
    }
    return all_signs.get(sign_code, sign_code)

def get_sign_full_display(sign_code: str) -> str:
    """
    Get full display string with code and name
    
    Args:
        sign_code: Sign code (e.g., 'P.102')
        
    Returns:
        Full display string (e.g., 'P.102: Cấm đi ngược chiều')
    """
    name = get_sign_name(sign_code)
    if name == sign_code:
        return sign_code
    return f"{sign_code}: {name}"

def get_sign_category(sign_code: str) -> str:
    """
    Get category of a sign based on its code
    
    Args:
        sign_code: Sign code (e.g., 'P.102')
        
    Returns:
        Category name in Vietnamese
    """
    if sign_code.startswith('P.'):
        return 'Biển cấm'
    elif sign_code.startswith('W.'):
        return 'Biển cảnh báo'
    elif sign_code.startswith('R.'):
        return 'Biển hiệu lệnh'
    elif sign_code.startswith('I.') or sign_code.startswith('S.'):
        return 'Biển chỉ dẫn'
    elif sign_code.startswith('DP.'):
        return 'Biển phụ'
    else:
        return 'Khác'


# Biển báo HIỆU LỆNH (R - Mandatory Signs)
MANDATORY_SIGNS = {
    "R.122": "Dừng lại",
    "R.301a": "Chỉ được đi thẳng",
    "R.301c": "Chỉ được rẽ trái",
    "R.301d": "Chỉ được đi thẳng hoặc rẽ phải",
    "R.301e": "Chỉ được rẽ trái hoặc rẽ phải",
    "R.302a": "Phải đi vòng chướng ngại vật bên phải",
    "R.302b": "Phải đi vòng chướng ngại vật bên trái",
    "R.303": "Nơi giao nhau chạy theo vòng xuyến",
    "R.306": "Tốc độ tối thiểu cho phép",
    "R.407a": "Đường một chiều",
    "R.409": "Chỗ quay xe",
    "R.415a": "Đường dành cho ô tô",
    "R.425": "Bệnh viện",
    "R.434": "Bến xe buýt",
}

# Biển báo CHỈ DẪN (I/S - Information Signs)
INFORMATION_SIGNS = {
    "S.505a": "Hướng đi trên các làn đường",
    "S.509a": "Hướng đi trên các làn đường tại nút giao",
    "DP.135": "Hết tất cả các lệnh cấm",
}

# ============================================================================
# ALL SIGNS COMBINED
# ============================================================================
ALL_SIGNS = {
    **PROHIBITORY_SIGNS,
    **WARNING_SIGNS,
    **MANDATORY_SIGNS,
    **INFORMATION_SIGNS
}
