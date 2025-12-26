<template>
  <main class="main-content detect-page">
    <div class="detect-layout">

      <div class="panel control-panel">
        <h3>Bảng điều khiển</h3>
        <div class="controls-content">
          <div class="control-group">
            <label for="modelType">Loại Model:</label>
            <select id="modelType" v-model="modelType">
              <option value="yolo">YOLO (Detection)</option>
              <option value="cnn">CNN (Classification)</option>
            </select>
          </div>

          <div class="control-group">
            <label for="modelSelect">Chọn Model:</label>
            <select id="modelSelect" v-model="selectedModel">
              <option value="" disabled>
                {{ currentModelList.length > 0 ? 'Chọn một model' : 'Đang tải...' }}
              </option>
              <option v-for="model in currentModelList" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
          </div>

          <div class="control-group file-upload-group">
             <label for="imageUpload" class="button-style file-label">
                <vue-feather type="upload" size="18"></vue-feather>
                <span>{{ imageFile ? imageFile.name : 'Chọn ảnh từ máy' }}</span>
             </label>
             <input type="file" id="imageUpload" @change="onFileChange" accept="image/*" style="display: none;">
          </div>
          
          <button @click="predict" class="primary predict-button" :disabled="!imageFile || !selectedModel || isLoading">
             <vue-feather v-if="!isLoading" type="zap" size="18"></vue-feather>
             <span v-if="isLoading" class="spinner"></span>
             {{ isLoading ? 'Đang nhận diện...' : 'Bắt đầu Nhận diện' }}
          </button>
        </div>
        <p class="status-text">{{ status }}</p>
      </div>

      <div class="panel display-panel">
        <div class="image-display-section">
          <h3>Xem trước Ảnh</h3>
          <div class="image-container" :class="{ 'has-image': imagePreviewUrl }">
            <img v-if="imagePreviewUrl" :src="imagePreviewUrl" id="imagePreview" alt="Ảnh gốc">
            <div v-else class="placeholder">
              <vue-feather type="image" size="48" class="placeholder-icon"></vue-feather>
              <p>Ảnh gốc sẽ hiển thị ở đây</p>
            </div>
            <canvas ref="canvasRef" id="overlayCanvas"></canvas>
          </div>
        </div>

        <div class="results-section">
            <h3>Kết quả Phát hiện</h3>
            <div v-if="isLoading" class="loading-results">
              <span class="spinner"></span> Đang phân tích...
            </div>
            <div v-else-if="predictionsDone && results.length > 0" class="results-list">
              <ul>
                <li v-for="(result, index) in results" :key="index">
                  <span class="result-index">{{ index + 1 }}.</span>
                  <span class="result-name">{{ result.class_name }}</span>
                  <span class="result-confidence">
                    ({{ ((result.confidence || result.cnn_confidence || 0) * 100).toFixed(0) }}%)
                  </span>
                </li>
              </ul>
            </div>
            <div v-else-if="predictionsDone && results.length === 0" class="no-results">
              <vue-feather type="alert-circle" size="20"></vue-feather>
              Không phát hiện được biển báo nào.
            </div>
             <div v-else class="no-results idle">
               Kết quả sẽ hiển thị ở đây sau khi nhận diện.
            </div>
        </div>
      </div>

    </div>
  </main>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import VueFeather from 'vue-feather';

// State
const modelType = ref('yolo');
const yoloModels = ref([]);
const cnnModels = ref([]);
const selectedModel = ref('');
const status = ref('Vui lòng chọn model và ảnh.');
const imagePreviewUrl = ref(null);
const imageFile = ref(null);
const currentImage = ref(null);
const results = ref([]);
const predictionsDone = ref(false);
const isLoading = ref(false);

// Refs
const canvasRef = ref(null);

// Computed
const currentModelList = computed(() => {
  return modelType.value === 'yolo' ? yoloModels.value : cnnModels.value;
});

// Lifecycle Hooks
onMounted(() => {
  loadModels();
});

// Watch modelType để reset selectedModel
watch(modelType, () => {
  selectedModel.value = '';
  if (currentModelList.value.length > 0) {
    selectedModel.value = currentModelList.value[0];
  }
});

// Methods
async function loadModels() {
  isLoading.value = true;
  status.value = 'Đang tải danh sách model...';
  try {
    const response = await fetch('http://localhost:8000/models');
    if (!response.ok) throw new Error('Không thể tải model');
    const models = await response.json();
    
    yoloModels.value = models.yolo || [];
    cnnModels.value = models.cnn || [];
    
    if (yoloModels.value.length > 0) {
      selectedModel.value = yoloModels.value[0];
    }
    
    status.value = 'Sẵn sàng! Vui lòng chọn model và ảnh.';
  } catch (error) {
    console.error('Lỗi tải models:', error);
    status.value = `Lỗi kết nối backend: ${error.message}`;
  } finally {
     isLoading.value = false;
  }
}

function onFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  imageFile.value = file;
  results.value = [];
  predictionsDone.value = false;
  imagePreviewUrl.value = null;
  clearCanvas();

  const reader = new FileReader();
  reader.onload = (event) => {
    imagePreviewUrl.value = event.target.result;
    
    const img = new Image();
    img.onload = () => {
      currentImage.value = img;
      // Resize canvas khớp với ảnh
      const canvas = canvasRef.value;
      if (canvas) {
          canvas.width = img.width;
          canvas.height = img.height;
      }
      status.value = `Đã chọn ảnh: ${file.name}. Nhấn "Bắt đầu Nhận diện".`;
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

async function predict() {
  if (!selectedModel.value || !imageFile.value || !currentImage.value) {
    status.value = 'Lỗi: Vui lòng chọn model và ảnh hợp lệ.';
    return;
  }

  isLoading.value = true;
  status.value = `Đang xử lý với model ${selectedModel.value}...`;
  results.value = [];
  predictionsDone.value = false;
  clearCanvas();

  const formData = new FormData();
  formData.append('file', imageFile.value);
  formData.append('model_name', selectedModel.value);
  formData.append('model_type', modelType.value);

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Lỗi server: ${errData.detail || response.statusText}`);
    }

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    results.value = data.predictions;
    drawResults(data.predictions);
    
    status.value = `Hoàn tất! Tìm thấy ${data.predictions.length} đối tượng.`;

  } catch (error) {
    console.error('Lỗi nhận diện:', error);
    status.value = `Đã xảy ra lỗi: ${error.message}`;
  } finally {
    isLoading.value = false;
    predictionsDone.value = true;
  }
}

function drawResults(predictions) {
  const canvas = canvasRef.value;
  if (!canvas || !currentImage.value) return;
  
  const ctx = canvas.getContext('2d');
  // Đảm bảo canvas có kích thước đúng bằng ảnh
  canvas.width = currentImage.value.width;
  canvas.height = currentImage.value.height;

  // Xóa canvas trước khi vẽ
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!predictions) return;

  predictions.forEach(p => {
    let x1, y1, x2, y2;

    // --- LOGIC QUAN TRỌNG: XỬ LÝ TOẠ ĐỘ ---
    if (p.box_coordinates) {
        // YOLO: Có toạ độ thật
        [x1, y1, x2, y2] = p.box_coordinates;
    } else {
        // CNN: Không có toạ độ -> Lấy full ảnh
        x1 = 10;
        y1 = 10;
        x2 = canvas.width - 10;
        y2 = canvas.height - 10;
    }

    const confidence = p.confidence || p.cnn_confidence || 0;
    const className = p.class_name || "Unknown";

    // Màu sắc: Xanh dương (YOLO), Xanh lá (CNN)
    const color = p.box_coordinates ? '#3b82f6' : '#10b981';

    // Vẽ bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Vẽ label text
    const text = `${className} (${(confidence * 100).toFixed(0)}%)`;
    ctx.font = 'bold 18px "Inter", sans-serif';
    const textMetrics = ctx.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = 18;

    // Vị trí label
    const labelX = x1;
    // Nếu là YOLO, vẽ nhãn lên trên. Nếu CNN (full ảnh), vẽ nhãn vào trong góc.
    const labelY = p.box_coordinates ? (y1 - 10) : (y1 + 30);

    // Vẽ nền cho label
    ctx.fillStyle = color;
    ctx.fillRect(labelX - 1, labelY - textHeight - 6, textWidth + 12, textHeight + 10);

    // Vẽ chữ
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, labelX + 5, labelY - 6);
  });
}

function clearCanvas() {
   const canvas = canvasRef.value;
   if(canvas){
       const ctx = canvas.getContext('2d');
       ctx.clearRect(0, 0, canvas.width, canvas.height);
   }
}
</script>

<style scoped>
/* Layout 2 cột */
.detect-layout {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 32px;
  align-items: flex-start;
}

/* Panel điều khiển */
.control-panel {
  position: sticky;
  top: 32px;
  max-width: 380px;
  overflow: hidden;
}

.controls-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.control-group label {
  font-weight: 600;
  color: #2d3748;
  font-size: 0.95rem;
  letter-spacing: -0.025em;
}

.control-group select {
  padding: 12px 14px;
  background: #fff;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  color: #2d3748;
  font-weight: 500;
  transition: all 0.3s ease;
}

.control-group select:focus {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Nút chọn file tùy chỉnh */
.file-upload-group {
  margin-top: 8px;
}

.file-label {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
  color: #4a5568;
  border: 2px dashed #cbd5e0;
  cursor: pointer;
  padding: 16px 18px;
  border-radius: 10px;
  transition: all 0.3s ease;
  font-weight: 600;
}

.file-label:hover {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-color: #667eea;
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
}

.file-label span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  flex: 1;
  font-size: 0.9rem;
}

/* Nút nhận diện */
.predict-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-top: 12px;
  padding: 14px 18px;
  font-size: 1rem;
}

.predict-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Trạng thái */
.status-text {
  margin-top: 24px;
  padding: 14px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
  border-left: 4px solid #667eea;
  border-radius: 6px;
  color: #4a5568;
  font-size: 0.85rem;
  text-align: left;
  min-height: 1.2em;
  font-weight: 500;
  word-wrap: break-word;
  overflow-wrap: break-word;
  max-width: 100%;
  line-height: 1.5;
}

/* Panel hiển thị */
.display-panel {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

/* Khu vực ảnh */
.image-display-section h3 {
  margin-bottom: 16px;
}

.image-container {
  position: relative;
  width: fit-content; /* Shrink to fit image */
  margin: 0 auto;     /* Center horizontally */
  min-width: 300px;
  min-height: 200px;
  max-height: 80vh;
  max-width: 100%;    /* Don't overflow screen */
  background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: all 0.3s ease;
}

.image-container:hover {
  border-color: #cbd5e0;
}

.image-container.has-image {
  background: #000;
}

#imagePreview {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

#overlayCanvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* Placeholder */
.placeholder {
  text-align: center;
  color: #a0aec0;
}

.placeholder-icon {
  margin-bottom: 12px;
  color: #cbd5e0;
  opacity: 0.6;
}

.placeholder p {
  font-weight: 500;
  font-size: 0.95rem;
}

/* Khu vực kết quả */
.results-section h3 {
  margin-bottom: 16px;
}

.results-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
  max-height: 280px;
  overflow-y: auto;
  border: 2px solid #e2e8f0;
  border-radius: 10px;
  background: #fff;
}

.results-list ul::-webkit-scrollbar {
  width: 8px;
}

.results-list ul::-webkit-scrollbar-track {
  background: #f7fafc;
}

.results-list ul::-webkit-scrollbar-thumb {
  background: #cbd5e0;
  border-radius: 4px;
}

.results-list li {
  padding: 14px 18px;
  border-bottom: 1px solid #edf2f7;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 0.95rem;
  transition: background 0.2s ease;
}

.results-list li:hover {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
}

.results-list li:last-child {
  border-bottom: none;
}

.result-index {
  color: #a0aec0;
  font-weight: 600;
  min-width: 24px;
}

.result-name {
  color: #2d3748;
  font-weight: 700;
  flex-grow: 1;
}

.result-confidence {
  color: #667eea;
  font-weight: 700;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.85rem;
}

/* Thông báo */
.no-results, .loading-results {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 32px;
  border: 2px dashed #e2e8f0;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.02) 0%, rgba(118, 75, 162, 0.02) 100%);
  color: #718096;
  min-height: 120px;
  font-weight: 500;
}

.no-results.idle {
  color: #a0aec0;
  font-style: italic;
}

/* Spinner */
.spinner {
  width: 18px;
  height: 18px;
  border: 3px solid currentColor;
  border-bottom-color: transparent;
  border-radius: 50%;
  display: inline-block;
  box-sizing: border-box;
  animation: spinner-rotation 0.8s linear infinite;
}

@keyframes spinner-rotation {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>