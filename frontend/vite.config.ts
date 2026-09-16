import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // 连续写入合并后再热更新；WSL 使用轮询，避免漏掉文件事件后继续返回旧模块。
    watch: {
      usePolling: Boolean(process.env.WSL_DISTRO_NAME),
      interval: 300,
      awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 100 },
    },
    port: Number(process.env.MOONLIGHTBOX_FRONTEND_PORT ?? '5175'),
    proxy: {
      '/api': `http://127.0.0.1:${process.env.MOONLIGHTBOX_BACKEND_PORT ?? '8001'}`,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
