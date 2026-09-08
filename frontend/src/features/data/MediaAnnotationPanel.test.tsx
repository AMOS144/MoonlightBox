import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { TestThemeProvider } from '../../test/TestThemeProvider'
import { MediaAnnotationPanel } from './MediaAnnotationPanel'

afterEach(() => {
  vi.unstubAllGlobals()
})

test('明确批准时保留原语义证据并提交复用决定', async () => {
  const writes: Array<Record<string, unknown>> = []
  const annotation = {
    id: 'annotation-1',
    asset_id: 'audio-1',
    modality: 'audio',
    status: 'succeeded',
    summary: '',
    transcript: '我刚到家，你呢',
    ocr_text: '',
    safety_tags: [],
    source_model: 'mlx-community/whisper-large-v3-turbo',
    source_version: '0.4.3',
    confidence: 0.91,
    reusable: false,
    reuse_decision: 'pending',
    failure_code: null,
  }
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        writes.push(JSON.parse(String(init.body)) as Record<string, unknown>)
        return Promise.resolve(
          new Response(
            JSON.stringify({
              ...annotation,
              reusable: true,
              reuse_decision: 'approved',
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      const body = path.endsWith('/summary')
        ? {
            eligible: 2,
            annotated: 1,
            succeeded: 1,
            failed: 0,
            needs_review: 0,
            approved: 0,
            blocked: 0,
          }
        : [annotation]
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )

  render(
    <TestThemeProvider>
      <MediaAnnotationPanel projectId="project-1" />
    </TestThemeProvider>,
  )

  expect(await screen.findByText('我刚到家，你呢')).toBeInTheDocument()
  expect(screen.getByText('总进度 1/2')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '批准复用' }))

  await waitFor(() => {
    expect(writes).toEqual([
      expect.objectContaining({
        transcript: '我刚到家，你呢',
        confidence: 0.91,
        reuse_decision: 'approved',
      }),
    ])
  })
})
