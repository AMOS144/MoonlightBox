import { useState } from 'react'

type Preview = {
  id: string
  message_count: number
  participants: string[]
  time_range: [string, string] | null
  kind_counts: Record<string, number>
  sample_messages: Array<{
    timestamp: string
    sender: string
    kind: string
    content: string
  }>
  errors: Array<{ line: number; code: string; message: string }>
}

type ImportWizardProps = {
  projectId: string
}

export function ImportWizard({ projectId }: ImportWizardProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [selfParticipant, setSelfParticipant] = useState('')
  const [targetParticipant, setTargetParticipant] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function scan() {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(`/api/projects/${projectId}/imports/preview`, {
      method: 'POST',
      body: form,
    })
    setPreview((await response.json()) as Preview)
  }

  async function confirm() {
    if (!preview) return
    setSubmitting(true)
    try {
      await fetch(`/api/projects/${projectId}/imports/${preview.id}/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          self_participant: selfParticipant,
          target_participant: targetParticipant,
        }),
      })
    } finally {
      setSubmitting(false)
    }
  }

  const canConfirm =
    preview !== null &&
    selfParticipant !== '' &&
    targetParticipant !== '' &&
    selfParticipant !== targetParticipant

  return (
    <section className="import-wizard">
      <p className="eyebrow">导入数据</p>
      <h1>带我回到那一天</h1>
      <label htmlFor="chat-file">聊天记录文件</label>
      <input
        accept=".csv,.json,.txt"
        id="chat-file"
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        type="file"
      />
      <button className="primary-button" disabled={!file} onClick={scan} type="button">
        扫描并预览
      </button>

      {preview && (
        <div className="preview-panel">
          <h2>共 {preview.message_count} 条消息</h2>
          <label htmlFor="self-participant">我</label>
          <select
            id="self-participant"
            onChange={(event) => setSelfParticipant(event.target.value)}
            value={selfParticipant}
          >
            <option value="">请选择</option>
            {preview.participants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <label htmlFor="target-participant">复刻对象</label>
          <select
            id="target-participant"
            onChange={(event) => setTargetParticipant(event.target.value)}
            value={targetParticipant}
          >
            <option value="">请选择</option>
            {preview.participants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <button
            className="primary-button"
            disabled={!canConfirm || submitting}
            onClick={confirm}
            type="button"
          >
            {submitting ? '正在导入……' : '确认导入'}
          </button>
        </div>
      )}
    </section>
  )
}
