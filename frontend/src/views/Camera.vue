<template>
    <main class="main-content">
        <h1>Nhận diện Biển báo (Live Camera)</h1>
        
        <div class="camera-container">
            <div class="panel control-panel">
                <h3>Bảng điều khiển</h3>
                
                <div class="control-group">
                    <label>Trạng thái:</label>
                    <div class="connection-status" :class="statusClass">
                        {{ statusLabel }}
                    </div>
                </div>

                <div class="control-group">
                    <label>Chế độ:</label>
                    <select v-model="detectionMode" :disabled="isCameraOn">
                        <option value="yolo">YOLO Only (Nhanh)</option>
                        <option value="two-stage">YOLO + CNN (Chính xác)</option>
                    </select>
                </div>

                <div class="button-group">
                    <button v-if="!isCameraOn" @click="startCamera" class="primary">
                        ▶️ Bật Camera
                    </button>
                    <button v-if="isCameraOn" @click="stopCamera" class="secondary">
                        ⏹️ Tắt Camera
                    </button>
                    
                    <button 
                        v-if="isCameraOn" 
                        @click="takeSnapshot" 
                        class="snapshot-btn"
                        :disabled="isSnapshotting"
                    >
                        📸 {{ isSnapshotting ? 'Đang chụp...' : 'Chụp ảnh' }}
                    </button>
                </div>
                
                <p class="status-text">{{ status }}</p>

                <div class="results-list-mini" v-if="results.length > 0">
                    <h4>Phát hiện ({{ results.length }}):</h4>
                    <ul>
                        <li v-for="(res, idx) in results" :key="idx">
                            <strong>{{ res.class_name }}</strong> 
                            <small>({{ ((res.cnn_confidence || res.confidence) * 100).toFixed(0) }}%)</small>
                        </li>
                    </ul>
                </div>
            </div>
            
            <div class="panel video-panel">
                <h3>Stream từ Camera</h3>
                <div class="video-wrapper" :class="{ 'flash': isFlashing }">
                    <video ref="videoRef" id="video" autoplay playsinline muted></video>
                    <canvas ref="canvasRef" id="overlay-canvas"></canvas>
                    <div v-if="isProcessing" class="processing-indicator">
                        <span class="spinner"></span>
                    </div>
                </div>
                <div class="stats" v-if="isCameraOn">
                    <span>FPS: {{ fps }}</span>
                    <span>Objects: {{ results.length }}</span>
                    <span v-if="isProcessing" class="processing-badge">⏳ Đang xử lý...</span>
                </div>
            </div>
        </div>
    </main>
</template>

<script>
export default {
    data() {
        return {
            detectionMode: 'two-stage',
            selectedYoloModel: 'best.pt',
            selectedCnnModel: 'bien_bao_mobilenetv2_AUGMENTED_BALANCED_model.h5',
            
            status: 'Sẵn sàng. Nhấn "Bật Camera" để bắt đầu.',
            isCameraOn: false,
            stream: null,
            fps: 0,
            lastFrameTime: Date.now(),
            
            results: [],
            
            // Flow Control - Strict Mutex
            isProcessing: false,
            abortController: null,
            animationFrameId: null,
            
            // Snapshot feature
            isSnapshotting: false,
            isFlashing: false
        }
    },
    computed: {
        statusClass() {
            if (!this.isCameraOn) return 'disconnected';
            if (this.isProcessing) return 'processing';
            return 'connected';
        },
        statusLabel() {
            if (!this.isCameraOn) return '⚫ Camera đã tắt';
            if (this.isProcessing) return '🔄 Đang xử lý frame...';
            return '🟢 Camera đang chạy';
        }
    },
    beforeUnmount() {
        this.stopCamera();
    },
    methods: {
        // ============================================================
        // CAMERA CONTROL
        // ============================================================
        async startCamera() {
            try {
                this.status = 'Đang khởi động camera...';
                
                // Request camera with LOW resolution to reduce backend load
                this.stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 640, max: 640 }, 
                        height: { ideal: 480, max: 480 }, 
                        facingMode: 'user',
                        frameRate: { ideal: 15, max: 20 } // Limit FPS
                    }
                });
                
                const video = this.$refs.videoRef;
                video.srcObject = this.stream;
                
                await new Promise(resolve => video.onloadedmetadata = resolve);
                await video.play();
                
                // Setup Canvas to match video
                const canvas = this.$refs.canvasRef;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                this.isCameraOn = true;
                this.isProcessing = false;
                this.results = [];
                this.status = 'Camera đã bật. Đang nhận diện...';
                
                // Start detection loop
                this.loopDetect();
                
            } catch (err) {
                console.error('Camera error:', err);
                this.status = '❌ Lỗi camera: ' + err.message;
            }
        },

        stopCamera() {
            // 1. Stop the loop FIRST
            this.isCameraOn = false;
            
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
            
            // 2. ABORT any pending request immediately (Critical!)
            if (this.abortController) {
                this.abortController.abort();
                this.abortController = null;
                console.log('🛑 Aborted pending request');
            }
            
            // 3. Stop camera stream
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            
            // 4. Clear canvas
            const canvas = this.$refs.canvasRef;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            
            // 5. Reset all state
            this.results = [];
            this.isProcessing = false;
            this.fps = 0;
            this.status = 'Camera đã tắt.';
        },

        // ============================================================
        // DETECTION LOOP - Strict Flow Control
        // ============================================================
        loopDetect() {
            // Exit if camera is off
            if (!this.isCameraOn) return;

            // STRICT MUTEX: Only send new frame if NOT processing
            if (!this.isProcessing) {
                this.detectFrame();
            }

            // Schedule next iteration
            this.animationFrameId = requestAnimationFrame(() => this.loopDetect());
        },

        async detectFrame() {
            // Double-check camera is still on
            if (!this.isCameraOn) return;

            // LOCK - Prevent any new requests
            this.isProcessing = true;
            
            // Create new AbortController for this request
            this.abortController = new AbortController();
            const signal = this.abortController.signal;

            try {
                const video = this.$refs.videoRef;
                
                // Capture frame
                const captureCanvas = document.createElement('canvas');
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
                captureCanvas.getContext('2d').drawImage(video, 0, 0);

                // Compress to JPEG (0.5 quality for faster upload)
                const blob = await new Promise(resolve => 
                    captureCanvas.toBlob(resolve, 'image/jpeg', 0.5)
                );
                
                // Prepare request
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                let url = 'http://localhost:8000/predict';
                if (this.detectionMode === 'yolo') {
                    formData.append('model_name', this.selectedYoloModel);
                    formData.append('model_type', 'yolo');
                } else {
                    url = 'http://localhost:8000/predict_two_stage';
                    formData.append('yolo_model', this.selectedYoloModel);
                    formData.append('cnn_model', this.selectedCnnModel);
                    formData.append('confidence_threshold', 0.25);
                }

                // Send request with AbortController
                const res = await fetch(url, { 
                    method: 'POST', 
                    body: formData,
                    signal: signal
                });

                // Only process if camera is still on
                if (res.ok && this.isCameraOn) {
                    const data = await res.json();
                    
                    this.results = data.predictions || [];
                    this.drawResults(this.results);
                    
                    // Calculate FPS
                    const now = Date.now();
                    const delta = now - this.lastFrameTime;
                    if (delta > 0) {
                        this.fps = Math.round(1000 / delta);
                    }
                    this.lastFrameTime = now;
                }

            } catch (e) {
                if (e.name === 'AbortError') {
                    console.log('🛑 Request aborted (camera stopped or page changed)');
                } else {
                    console.error('Detection error:', e);
                    this.status = '⚠️ Lỗi: ' + e.message;
                }
            } finally {
                // UNLOCK - Allow next request
                this.isProcessing = false;
                this.abortController = null;
            }
        },

        // ============================================================
        // SNAPSHOT FEATURE
        // ============================================================
        async takeSnapshot() {
            if (!this.isCameraOn || this.isSnapshotting) return;
            
            this.isSnapshotting = true;
            
            try {
                const video = this.$refs.videoRef;
                
                // Flash effect
                this.isFlashing = true;
                setTimeout(() => { this.isFlashing = false; }, 150);
                
                // Capture high-quality frame
                const snapshotCanvas = document.createElement('canvas');
                snapshotCanvas.width = video.videoWidth;
                snapshotCanvas.height = video.videoHeight;
                const ctx = snapshotCanvas.getContext('2d');
                
                // Draw video frame
                ctx.drawImage(video, 0, 0);
                
                // Draw detection boxes on snapshot (optional)
                if (this.results.length > 0) {
                    this.drawResultsOnCanvas(ctx, this.results);
                }
                
                // Convert to high-quality JPEG blob
                const blob = await new Promise(resolve => 
                    snapshotCanvas.toBlob(resolve, 'image/jpeg', 0.95)
                );
                
                // Generate filename with timestamp
                const timestamp = new Date().toISOString()
                    .replace(/[:.]/g, '-')
                    .replace('T', '_')
                    .slice(0, 19);
                const filename = `detection_shot_${timestamp}.jpg`;
                
                // Trigger download
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
                
                this.status = `📸 Đã lưu: ${filename}`;
                
            } catch (e) {
                console.error('Snapshot error:', e);
                this.status = '❌ Lỗi chụp ảnh: ' + e.message;
            } finally {
                this.isSnapshotting = false;
            }
        },

        // ============================================================
        // DRAWING FUNCTIONS
        // ============================================================
        drawResults(predictions) {
            const canvas = this.$refs.canvasRef;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!predictions || predictions.length === 0) return;
            
            this.drawResultsOnCanvas(ctx, predictions);
        },
        
        drawResultsOnCanvas(ctx, predictions) {
            predictions.forEach(p => {
                if (!p.box_coordinates) return;
                
                const [x1, y1, x2, y2] = p.box_coordinates;
                const confidence = p.cnn_confidence || p.confidence || 0;
                
                // Draw bounding box
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Prepare label
                const text = `${p.class_name} (${(confidence * 100).toFixed(0)}%)`;
                ctx.font = "bold 14px Arial";
                const textWidth = ctx.measureText(text).width;
                
                // Draw label background
                ctx.fillStyle = "rgba(0, 255, 0, 0.85)";
                ctx.fillRect(x1, y1 - 22, textWidth + 10, 22);
                
                // Draw label text
                ctx.fillStyle = "#000";
                ctx.fillText(text, x1 + 5, y1 - 6);
            });
        }
    }
}
</script>


<style scoped>
.camera-container {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 24px;
}

.control-panel {
    position: sticky;
    top: 20px;
    max-height: calc(100vh - 120px);
    overflow-y: auto;
}

.control-group {
    margin-bottom: 16px;
}

.control-group label {
    display: block;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 8px;
    font-size: 0.9rem;
}

.control-group select {
    width: 100%;
    padding: 10px 12px;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 0.9rem;
    background: #fff;
}

.control-group select:disabled {
    background: #f7fafc;
    cursor: not-allowed;
}

.connection-status {
    padding: 10px 14px;
    text-align: center;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.connected {
    background: #c6f6d5;
    color: #22543d;
}

.disconnected {
    background: #fed7d7;
    color: #9b2c2c;
}

.processing {
    background: #fefcbf;
    color: #975a16;
}

.button-group {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 16px;
}

button {
    width: 100%;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.secondary {
    background: #e53e3e;
    color: white;
}

.secondary:hover:not(:disabled) {
    background: #c53030;
}

.snapshot-btn {
    background: linear-gradient(135deg, #38b2ac 0%, #319795 100%);
    color: white;
}

.snapshot-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(56, 178, 172, 0.4);
}

.status-text {
    margin-top: 16px;
    padding: 12px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border-left: 4px solid #667eea;
    border-radius: 6px;
    color: #4a5568;
    font-size: 0.85rem;
    font-weight: 500;
    word-break: break-word;
}

.results-list-mini {
    margin-top: 16px;
    background: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.results-list-mini h4 {
    margin: 0 0 10px 0;
    font-size: 0.9rem;
    color: #2d3748;
}

.results-list-mini ul {
    list-style: none;
    padding: 0;
    margin: 0;
    max-height: 200px;
    overflow-y: auto;
}

.results-list-mini li {
    padding: 8px 0;
    border-bottom: 1px solid #edf2f7;
    font-size: 0.85rem;
}

.results-list-mini li:last-child {
    border-bottom: none;
}

.video-panel {
    display: flex;
    flex-direction: column;
}

.video-wrapper {
    position: relative;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    width: fit-content; /* Fit to video size */
    margin: 0 auto;
    max-width: 100%;
    /* Removed aspect-ratio: 4/3 to avoid forcing ratio */
}

.video-wrapper.flash {
    filter: brightness(2);
}

#video {
    display: block;
    max-width: 100%;
    height: auto; /* Maintain aspect ratio */
    /* Remove object-fit to avoid letterboxing mismatch */
}

#overlay-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    /* Ensure canvas scales nicely if video scales */
}

.processing-indicator {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(0, 0, 0, 0.7);
    padding: 8px 12px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #fff;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.stats {
    display: flex;
    gap: 20px;
    margin-top: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border-radius: 8px;
}

.stats span {
    font-weight: 600;
    color: #2d3748;
    font-size: 0.9rem;
}

.processing-badge {
    color: #d69e2e !important;
    background: rgba(214, 158, 46, 0.1);
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.8rem !important;
}

/* Responsive */
@media (max-width: 900px) {
    .camera-container {
        grid-template-columns: 1fr;
    }
    
    .control-panel {
        position: static;
        max-height: none;
    }
}
</style>
