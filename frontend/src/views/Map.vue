<template>
    <main class="main-content-full">
        <div class="map-controls">
            <h3>Highway Guardian Map (TP.HCM)</h3>
            <p>Hiển thị các vị trí biển báo (dữ liệu giả lập)</p>
        </div>
        <div ref="mapRef" id="map"></div>
    </main>
</template>

<script>
export default {
    data() {
        return {
            map: null, // Lưu đối tượng Google Map
            // Dữ liệu giả lập cho 3 loại biển báo
            signData: [
                // 1. Biển cấm (Đỗ xe, Rẽ) - Ví dụ ở Q1
                { 
                    position: { lat: 10.7769, lng: 106.7009 }, // (Gần Nhà hát Lớn)
                    title: 'Biển Cấm: Cấm đỗ xe',
                    type: 'Cấm'
                },
                { 
                    position: { lat: 10.7797, lng: 106.6990 }, // (Gần Dinh Độc Lập)
                    title: 'Biển Cấm: Cấm rẽ trái',
                    type: 'Cấm'
                },
                // 2. Hiệu lệnh (Bắt buộc) - Ví dụ ở Q3
                { 
                    position: { lat: 10.7831, lng: 106.6917 }, // (Hồ Con Rùa)
                    title: 'Biển Hiệu Lệnh: Bắt buộc đi thẳng',
                    type: 'Hiệu Lệnh'
                },
                // 3. Nguy hiểm (Đường ray) - Ví dụ ở Phú Nhuận
                { 
                    position: { lat: 10.7968, lng: 106.6802 }, // (Đường ray cắt Lê Văn Sỹ)
                    title: 'Biển Nguy Hiểm: Giao nhau với đường sắt',
                    type: 'Nguy Hiểm'
                }
            ]
        }
    },
    mounted() {
        // Hàm này chạy khi component được tải
        this.initMap();
    },
    methods: {
        initMap() {
            // Tọa độ trung tâm TP. Hồ Chí Minh
            const hcmCity = { lat: 10.7769, lng: 106.7009 };
            
            // Khởi tạo bản đồ
            this.map = new google.maps.Map(this.$refs.mapRef, {
                center: hcmCity,
                zoom: 14,
            });

            // Tạo các marker (chấm đỏ)
            this.signData.forEach(sign => {
                const marker = new google.maps.Marker({
                    position: sign.position,
                    map: this.map,
                    title: sign.title,
                });

                // Tạo cửa sổ thông tin khi click vào marker
                const infoWindow = new google.maps.InfoWindow({
                    content: `<strong>${sign.type}</strong><p>${sign.title}</p>`
                });

                marker.addListener('click', () => {
                    infoWindow.open(this.map, marker);
                });
            });
        }
    }
}
</script>

<style scoped>
/* Chúng ta ghi đè lại class .main-content 
    nhưng vì 'scoped', nó chỉ áp dụng ở file này 
*/
.main-content-full {
    flex-grow: 1;
    overflow: hidden;
    padding: 0; /* Xóa padding */
    position: relative; /* Để đặt control lên trên */
}
#map {
    width: 100%;
    height: 100vh;
}
.map-controls {
    position: absolute;
    top: 20px;
    left: 20px;
    background: #fff;
    padding: 15px 20px;
    border-radius: 8px;
    z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.map-controls h3 {
    margin: 0 0 5px 0;
    color: #2c3e50;
}
</style>