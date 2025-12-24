<template>
  <div class="app-container" :class="{ 'login-layout': isLoginPage }">
    <Sidebar v-if="!isLoginPage" />
    
    <div class="page-wrapper">
      <Header v-if="!isLoginPage" />
      <router-view />
    </div>

  </div>
</template>

<script>
import Sidebar from './components/Sidebar.vue';
import Header from './components/Header.vue'; 

export default {
  components: {
    Sidebar,
    Header
  },
  computed: {
    isLoginPage() {
      return this.$route.name === 'Login';
    }
  }
}
</script>

<style>
/* === CSS GLOBAL === */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  height: 100vh;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f5f7fa;
}

.app-container {
  display: flex;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  background-attachment: fixed;
}

.app-container.login-layout {
  display: block;
}

.page-wrapper {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f7fa;
}

.main-content {
  flex-grow: 1;
  overflow-y: auto;
  padding: 32px;
  background-color: #f5f7fa;
}

.main-content::-webkit-scrollbar {
  width: 8px;
}

.main-content::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.main-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.main-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.main-content h1 {
  color: #1a202c;
  margin-bottom: 24px;
  font-size: 1.875rem;
  font-weight: 700;
  letter-spacing: -0.025em;
}

.panel {
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  padding: 28px;
  transition: all 0.3s ease;
}

.panel:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

.panel h3 {
  margin-bottom: 20px;
  color: #2d3748;
  border-bottom: 2px solid #e2e8f0;
  padding-bottom: 12px;
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.025em;
}

label.button-style, button, select {
  display: block;
  width: 100%;
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s ease;
  border: 1px solid #cbd5e0;
  font-size: 0.95rem;
  text-align: center;
  box-sizing: border-box;
  outline: none;
}

label.button-style {
  background-color: #edf2f7;
  color: #4a5568;
}

label.button-style:hover {
  background-color: #e2e8f0;
  border-color: #a0aec0;
}

select {
  background-color: #fff;
  color: #2d3748;
  cursor: pointer;
}

select:focus {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

button.primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  font-weight: 700;
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
}

button.primary:hover {
  background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
  box-shadow: 0 6px 8px rgba(102, 126, 234, 0.4);
  transform: translateY(-1px);
}

button.primary:active {
  transform: translateY(0);
}

button.secondary {
  background-color: #edf2f7;
  color: #4a5568;
  border: 1px solid #cbd5e0;
}

button.secondary:hover {
  background-color: #e2e8f0;
  border-color: #a0aec0;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none !important;
}
</style>