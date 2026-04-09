<template>
    <main class="main-content">
        <h1>Lịch sử Nhận diện</h1>
        
        <div v-if="loading" class="empty-state">
            <p>⏳ Đang tải dữ liệu...</p>
        </div>
        
        <div v-else-if="error" class="empty-state error-banner">
            <p>⚠️ {{ error }}</p>
        </div>
        
        <div v-else-if="historyData.length === 0" class="empty-state-card">
            <div class="empty-icon">📭</div>
            <h2>Chưa có dữ liệu</h2>
            <p>Hệ thống chưa ghi nhận biển báo giao thông nào. Hãy tải ảnh lên để bắt đầu.</p>
        </div>
        
        <div v-else class="panel">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Thời gian</th>
                    <tr>
                        <th>ID</th>
                        <th>Trạng thái</th>
                        <th>Độ tin cậy</th>
                        <th>Ảnh</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="item in historyData" :key="item.id">
                        <td>{{ item.id }}</td>
                        <td>{{ item.timestamp }}</td>
                        <td>
                            <span v-if="item.is_valid" class="badge badge-success">✅ Hợp lệ</span>
                            <span v-else class="badge badge-error">❌ Không hợp lệ</span>
                        </td>
                        <td>{{ (item.confidence * 100).toFixed(1) }}%</td>
                        <td>
                            <img :src="getImageUrl(item.image_path)" class="thumbnail" alt="thumbnail">
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </main>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const historyData = ref([])
const loading = ref(true)
const error = ref('')

// Assuming backend is at VITE_API_URL or localhost:8000
const getApiUrl = () => import.meta.env.VITE_API_URL || 'http://localhost:8000'

const getImageUrl = (path: string) => {
    if (path.startsWith('http')) return path
    return `${getApiUrl()}/uploads/${path}` // or just the backend URL if static paths configured
}

const fetchHistory = async () => {
    try {
        loading.value = true
        error.value = ''
        const url = `${getApiUrl()}/api/v1/history?limit=50`
        const res = await fetch(url)
        if (!res.ok) throw new Error('Failed to fetch history')
        const data = await res.json()
        historyData.value = data.records || []
    } catch (err: any) {
        error.value = err.message
    } finally {
        loading.value = false
    }
}

onMounted(() => {
    fetchHistory()
})
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
.empty-state {
    padding: 2rem;
    text-align: center;
    color: #64748b;
}
.empty-state-card {
    background: #1e293b;
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: 4rem 2rem;
    text-align: center;
    color: #94a3b8;
    margin-top: 1rem;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.empty-state-card h2 {
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}
.badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: bold;
}
.badge-success { background: #22c55e; color: white; }
.badge-error { background: #ef4444; color: white; }
</style>