// frontend/src/main.ts

import { createApp } from 'vue'
import App from './App.vue'
import router from './router' // Đảm bảo đường dẫn đúng (./router/index.ts)
import { createPinia } from 'pinia'
import { useAuthStore } from './stores/authStore' // Đảm bảo đường dẫn đúng (./stores/authStore.ts)

// (Xóa import Toast và CSS của nó)
// (Xóa toastOptions)

// Khởi tạo App và Pinia
const app = createApp(App)
const pinia = createPinia()

// Cài đặt Pinia TRƯỚC Router
app.use(pinia)
// (Xóa app.use(Toast, toastOptions))

// Hàm bất đồng bộ để khởi động App
async function startApp() {
  const authStore = useAuthStore();
  await authStore.checkAuth();
  
  // Sau khi có trạng thái login, mới chạy Router và Mount App
  app.use(router)
  app.mount('#app')
}

// Gọi hàm khởi động
startApp();