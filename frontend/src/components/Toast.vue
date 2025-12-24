<template>
  <Transition name="toast">
    <div v-if="visible" :class="['toast', type]">
      <div class="toast-icon">
        <vue-feather v-if="type === 'success'" type="check-circle" size="24"></vue-feather>
        <vue-feather v-if="type === 'error'" type="alert-circle" size="24"></vue-feather>
        <vue-feather v-if="type === 'info'" type="info" size="24"></vue-feather>
      </div>
      <div class="toast-content">
        <h4 class="toast-title">{{ title }}</h4>
        <p class="toast-message">{{ message }}</p>
      </div>
      <button @click="close" class="toast-close">
        <vue-feather type="x" size="18"></vue-feather>
      </button>
    </div>
  </Transition>
</template>

<script setup>
import { ref, watch } from 'vue';
import VueFeather from 'vue-feather';

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  type: {
    type: String,
    default: 'info', // 'success', 'error', 'info'
    validator: (value) => ['success', 'error', 'info'].includes(value)
  },
  title: {
    type: String,
    default: ''
  },
  message: {
    type: String,
    default: ''
  },
  duration: {
    type: Number,
    default: 3000
  }
});

const emit = defineEmits(['close']);

let timer = null;

const close = () => {
  emit('close');
  if (timer) {
    clearTimeout(timer);
  }
};

watch(() => props.visible, (newVal) => {
  if (newVal && props.duration > 0) {
    timer = setTimeout(() => {
      close();
    }, props.duration);
  }
});
</script>

<style scoped>
.toast {
  position: fixed;
  top: 24px;
  right: 24px;
  min-width: 320px;
  max-width: 420px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.05);
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 18px 20px;
  z-index: 9999;
  backdrop-filter: blur(10px);
  animation: slideIn 0.3s ease-out;
}

.toast-icon {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.toast.success .toast-icon {
  background: linear-gradient(135deg, rgba(56, 161, 105, 0.15) 0%, rgba(72, 187, 120, 0.15) 100%);
  color: #38a169;
}

.toast.error .toast-icon {
  background: linear-gradient(135deg, rgba(229, 62, 62, 0.15) 0%, rgba(245, 101, 101, 0.15) 100%);
  color: #e53e3e;
}

.toast.info .toast-icon {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
  color: #667eea;
}

.toast-content {
  flex-grow: 1;
  padding-top: 2px;
}

.toast-title {
  font-size: 1rem;
  font-weight: 700;
  margin: 0 0 4px 0;
  color: #1a202c;
  letter-spacing: -0.025em;
}

.toast-message {
  font-size: 0.9rem;
  margin: 0;
  color: #718096;
  font-weight: 500;
  line-height: 1.4;
}

.toast-close {
  flex-shrink: 0;
  background: transparent;
  border: none;
  padding: 4px;
  cursor: pointer;
  color: #a0aec0;
  border-radius: 6px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  width: auto;
}

.toast-close:hover {
  background: #edf2f7;
  color: #4a5568;
}

/* Animations */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100px) scale(0.9);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100px) scale(0.9);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(100px) scale(0.9);
  }
  to {
    opacity: 1;
    transform: translateX(0) scale(1);
  }
}

/* Responsive */
@media (max-width: 640px) {
  .toast {
    top: 16px;
    right: 16px;
    left: 16px;
    min-width: auto;
    max-width: none;
  }
}
</style>
