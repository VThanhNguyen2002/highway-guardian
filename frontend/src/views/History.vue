<template>
    <main class="main-content">
        <h1>Lịch sử Nhận diện</h1>
        
        <div class="panel">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Thời gian</th>
                        <th>Ảnh (Thumbnail)</th>
                        <th>Kết quả phát hiện</th>
                        <th>Model đã dùng</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="item in historyData" :key="item.id">
                        <td>{{ item.id }}</td>
                        <td>{{ item.timestamp }}</td>
                        <td>
                            <img :src="item.imageUrl" class="thumbnail" alt="thumbnail">
                        </td>
                        <td>
                            <ul class="detections-list">
                                <li v-for="(detection, i) in item.detections" :key="i">
                                    {{ detection }}
                                </li>
                            </ul>
                        </td>
                        <td>{{ item.model }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </main>
</template>

<script>
export default {
    data() {
        return {
            // Dữ liệu giả (mock data)
            // Sau này, bạn sẽ fetch từ API backend
            historyData: [
                {
                    id: 1,
                    timestamp: '25/10/2025 14:30:15',
                    imageUrl: 'https://i.imgur.com/g0PNaAH.jpeg', // Ảnh mẫu
                    detections: ['Cấm rẽ trái (95%)', 'Giới hạn tốc độ (88%)'],
                    model: 'best_vietnam.pt'
                },
                {
                    id: 2,
                    timestamp: '24/10/2025 09:15:02',
                    imageUrl: 'https://i.imgur.com/7gK1gqf.jpeg', // Ảnh mẫu
                    detections: ['Bắt buộc đi thẳng (99%)'],
                    model: 'yolov8n_mapillary.pt'
                },
                {
                    id: 3,
                    timestamp: '23/10/2025 17:45:50',
                    imageUrl: 'https://i.imgur.com/rLZOaXF.jpeg', // Ảnh mẫu
                    detections: ['Cấm đỗ xe (92%)'],
                    model: 'best_vietnam.pt'
                }
            ]
        }
    }
}
</script>

<style scoped>
.history-table {
    width: 100%;
    border-collapse: collapse;
}
.history-table th, .history-table td {
    padding: 12px 15px;
    border: 1px solid #eee;
    text-align: left;
    vertical-align: middle;
}
.history-table th {
    background-color: #f9f9f9;
    font-weight: bold;
}
.thumbnail {
    width: 100px;
    height: auto;
    border-radius: 4px;
}
.detections-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.detections-list li {
    background: #f7f7f7;
    padding: 5px;
    border-radius: 3px;
    margin-bottom: 3px;
    font-size: 0.9rem;
}
</style>