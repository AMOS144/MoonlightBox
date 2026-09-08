import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
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
