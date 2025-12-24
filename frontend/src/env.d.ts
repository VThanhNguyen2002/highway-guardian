/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// Báo cho TS biết cách xử lý file .js
declare module '*.js' {
  const value: any;
  export default value;
}
