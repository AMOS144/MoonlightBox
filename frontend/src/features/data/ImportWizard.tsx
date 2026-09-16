import { useEffect, useRef, useState } from 'react'
import { Alert, Button, Select, Paper, Stack, Text, Title, VisuallyHidden } from '@mantine/core'
import { Link, useInRouterContext } from 'react-router-dom'

import { useSessionState } from '../../hooks/useSessionState'
import { Icon } from '../../components/Icon'
import { SetupHeader } from '../../components/SetupHeader'
import { mediaNotice, userMessage } from '../../components/feedback/messages'

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
  media_file_count: number
  linked_sticker_count: number
  unlinked_sticker_count: number
  deduplicated_asset_count?: number
  avatar_status?: Record<string, boolean>
  failure_reasons?: Record<string, number>
}

type ImportWizardProps = {
  projectId: string
  stage?: 'import' | 'participants'
  onSaved?: () => void
  savedImports?: Array<{ id: string; preview_id: string; message_count: number }>
}

type ConfirmResult = {
  import_id: string
  message_count: number
  created: boolean
  analysis_job_id: string | null
  world_job_id: string | null
}

export function ImportWizard({ projectId, stage = 'import', onSaved, savedImports = [] }: ImportWizardProps) {
  const inRouter = useInRouterContext()
  const directoryInput = useRef<HTMLInputElement | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [mediaFiles, setMediaFiles] = useState<File[]>([])
  const [directoryName, setDirectoryName] = useState('')
  const [preview, setPreview] = useSessionState<Preview | null>(`import-preview:${projectId}`, null)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState('')
  const [selfParticipant, setSelfParticipant] = useSessionState(`import-self:${projectId}`, '')
  const [targetParticipant, setTargetParticipant] = useSessionState(`import-target:${projectId}`, '')
  const [submitting, setSubmitting] = useState(false)
  const [savedConfirmation, setConfirmed] = useSessionState<ConfirmResult | null>(`import-confirmed:${projectId}:${preview?.id ?? 'none'}`, null)
  // 预览与保存状态必须绑定同一批导入；后端记录也可恢复旧版本页面未缓存的确认。
  const savedImport = savedImports.find(item => item.preview_id === preview?.id)
  const confirmed = savedImport ?? savedConfirmation
  const [confirmError, setConfirmError] = useState('')
  const scanController = useRef<AbortController | null>(null)
  const confirmController = useRef<AbortController | null>(null)
  const scanRequestId = useRef(0)
  const flowGeneration = useRef(0)

  useEffect(
    () => () => {
      flowGeneration.current += 1
      scanController.current?.abort()
      confirmController.current?.abort()
    },
    [],
  )

  async function scan() {
    if (!file || scanning) return
    scanController.current?.abort()
    const controller = new AbortController()
    scanController.current = controller
    const requestId = ++scanRequestId.current
    setScanning(true)
    setScanError('')
    const form = new FormData()
    form.append('file', file)
    mediaFiles.forEach((mediaFile) => {
      form.append('media_files', mediaFile)
      form.append(
        'media_paths',
        mediaFile.webkitRelativePath || mediaFile.name,
      )
    })
    try {
      const response = await fetch(`/api/projects/${projectId}/imports/preview`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
      if (!response.ok) throw new Error('扫描请求失败')
      const result = (await response.json()) as Preview
      if (requestId === scanRequestId.current && !controller.signal.aborted) {
        setPreview(result)
        setSelfParticipant((current) =>
          result.participants.includes(current) ? current : '',
        )
        setTargetParticipant((current) =>
          result.participants.includes(current) ? current : '',
        )
      }
    } catch (error) {
      if (
        requestId === scanRequestId.current &&
        !(error instanceof DOMException && error.name === 'AbortError')
      ) {
        setScanError('扫描失败，请检查文件后重试。')
      }
    } finally {
      if (requestId === scanRequestId.current) {
        setScanning(false)
      }
    }
  }

  async function confirm() {
    if (!preview || submitting) return
    confirmController.current?.abort()
    const controller = new AbortController()
    confirmController.current = controller
    const generation = flowGeneration.current
    setSubmitting(true)
    setConfirmError('')
    try {
      const response = await fetch(
        `/api/projects/${projectId}/imports/${preview.id}/confirm`,
        {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          self_participant: selfParticipant,
          target_participant: targetParticipant,
        }),
        },
      )
      if (!response.ok) {
        throw new Error('导入请求失败')
      }
      const result = (await response.json()) as ConfirmResult
      if (generation !== flowGeneration.current || controller.signal.aborted) return
      setConfirmed(result)
      onSaved?.()
    } catch (error) {
      if (
        generation === flowGeneration.current &&
        !(error instanceof DOMException && error.name === 'AbortError')
      ) {
        setConfirmError('导入失败，请检查文件和角色选择后重试。')
      }
    } finally {
      if (generation === flowGeneration.current) {
        setSubmitting(false)
      }
    }
  }

  // 已打开的页面可能仍持有旧版预览缓存；同样排除系统发送者，不要求用户重新上传。
  const personNames = (preview?.participants ?? []).filter(name => !['系统', 'system'].includes(name.trim().toLowerCase()))
  function selectPerson(role: 'self' | 'target', value: string | null) {
    if (submitting || !value || !personNames.includes(value)) return
    const current = role === 'self' ? selfParticipant : targetParticipant
    const other = role === 'self' ? targetParticipant : selfParticipant
    const setCurrent = role === 'self' ? setSelfParticipant : setTargetParticipant
    const setOther = role === 'self' ? setTargetParticipant : setSelfParticipant
    setCurrent(value)
    // 两人聊天自动配对；多人记录只在选择冲突时交换，不擅自猜测第三人的身份。
    if (personNames.length === 2) setOther(personNames.find(name => name !== value)!)
    else if (other === value) setOther(current !== value && personNames.includes(current) ? current : '')
  }
  const canConfirm =
    preview !== null &&
    personNames.includes(selfParticipant) &&
    personNames.includes(targetParticipant) &&
    selfParticipant !== targetParticipant

  function selectExportDirectory(files: File[]) {
    flowGeneration.current += 1
    scanController.current?.abort()
    confirmController.current?.abort()
    scanRequestId.current += 1
    const chatFiles = files.filter((candidate) =>
      /^chat\.(csv|json|txt)$/i.test(candidate.name),
    )
    const priority = ['chat.csv', 'chat.json', 'chat.txt']
    const selectedChat =
      priority
        .map((name) =>
          chatFiles.find((candidate) => candidate.name.toLowerCase() === name),
        )
        .find((candidate) => candidate !== undefined) ?? null
    const firstPath = files[0]?.webkitRelativePath
    setDirectoryName(firstPath?.split('/')[0] || '')
    setFile(selectedChat)
    setMediaFiles(
      files.filter(
        (candidate) =>
          !chatFiles.includes(candidate) &&
          candidate.name.toLowerCase() !== 'manifest.json' &&
          !candidate.name.startsWith('.'),
      ),
    )
    setScanning(false)
    setSubmitting(false)
    setPreview(null)
    setScanError(
      files.length > 0 && selectedChat === null
        ? '没有找到 chat.csv、chat.json 或 chat.txt，请选择完整的 WxEcho 导出目录。'
        : '',
    )
    setConfirmed(null)
    setConfirmError('')
    setSelfParticipant('')
    setTargetParticipant('')
  }

  return (
    <Stack className="setup-page" gap="lg">
      <SetupHeader step={stage === 'import' ? 1 : 2} title={stage === 'import' ? '导入记录' : '确认人物'}
        description={stage === 'import' ? '选择 WxEcho 导出目录，先解析预览，确认人物后才正式保存。' : '确认聊天中的你和目标人物。两者不能选择同一个人。'} />
      {stage === 'import' && <>
        <Paper p="lg" withBorder>
          <div className="setup-source">
            <Icon name="folder" size={24} />
            <div className="setup-source-copy">
              <Text fw={500}>{directoryName || (preview ? '已恢复解析预览' : '聊天记录目录')}</Text>
              <Text size="xs" c="dimmed">{file ? file.name + ' · ' + mediaFiles.length + ' 个媒体文件' : '支持 chat.csv、chat.json、chat.txt 及随附媒体'}</Text>
            </div>
            <Button variant="default" size="sm" disabled={scanning} onClick={() => directoryInput.current?.click()}>{preview ? '更换目录' : '选择目录'}</Button>
          </div>
        </Paper>
        <VisuallyHidden><input aria-label="WxEcho 导出目录" id="export-directory" multiple tabIndex={-1}
          onChange={event => selectExportDirectory(Array.from(event.target.files ?? []))}
          ref={element => { directoryInput.current = element; element?.setAttribute('webkitdirectory', '') }} type="file" /></VisuallyHidden>
      </>}
      {scanError && <Alert color="red" role="alert">{scanError}</Alert>}
      {stage === 'import' && !preview && <div className="setup-actions"><Button size="sm" disabled={!file || scanning} loading={scanning} onClick={scan} type="button">{scanning ? '正在扫描……' : '扫描并预览'}</Button></div>}
      {!preview && stage === 'participants' && <Alert>请先解析聊天记录，再确认双方身份。{inRouter && <div className="setup-actions"><Button component={Link} to={'/projects/' + projectId + '/setup/import'} variant="light">去导入记录</Button></div>}</Alert>}
      {preview && <>
        <section>
          <Title order={3}>{preview.message_count.toLocaleString('zh-CN')} 条消息{confirmed ? ' · 已保存' : ' · 已解析，尚未保存'}</Title>
          <div className="setup-summary">
            {preview.time_range && <Text size="sm" c="dimmed">{preview.time_range.map(t => new Date(t).toLocaleString('zh-CN')).join(' — ')}</Text>}
            <Text size="sm" c="dimmed">{personNames.join('、')}</Text>
            {preview.media_file_count > 0 && <Text size="sm" c="dimmed">{preview.media_file_count} 个媒体文件</Text>}
          </div>
          {Object.entries(preview.avatar_status ?? {}).filter(([, linked]) => !linked).map(([name]) => <Text key={name} size="sm" c="dimmed" mt="sm">{name}：未找到关联头像，不影响导入。</Text>)}
          {Object.entries(preview.failure_reasons ?? {}).filter(([, count]) => count > 0).map(([reason, count]) => <Alert color="moon" role="status" mt="sm" key={reason}>{mediaNotice(reason, count)}</Alert>)}
          {preview.errors?.length > 0 && <Alert color="orange" mt="sm">有 {preview.errors.length} 条记录需要检查。<details><summary>查看记录位置</summary>{preview.errors.map((e, i) => <Text key={i} size="sm">第 {e.line} 行：{userMessage(e.message)}</Text>)}</details></Alert>}
          {preview.media_file_count > 0 && <details><summary>媒体解析详情</summary><Text size="sm" c="dimmed">已关联 {preview.linked_sticker_count} 个表情，{preview.unlinked_sticker_count} 个待关联；去重后 {preview.deduplicated_asset_count ?? 0} 个资源。</Text></details>}
        </section>
        {(stage === 'participants' || !inRouter) && !confirmed && <div className="setup-people">
          <Select aria-label="我" id="self-participant" label="我" description="记录中由你发送的消息"
            placeholder="选择你的名字" value={personNames.includes(selfParticipant) ? selfParticipant : null}
            onChange={value => selectPerson('self', value)} disabled={submitting} allowDeselect={false}
            data={personNames} />
          <Select aria-label="目标人物" id="target-participant" label="目标人物" description="你想继续与谁聊天"
            placeholder="选择对方的名字" value={personNames.includes(targetParticipant) ? targetParticipant : null}
            onChange={value => selectPerson('target', value)} disabled={submitting} allowDeselect={false}
            data={personNames} />
          {selfParticipant && selfParticipant === targetParticipant && <Text size="sm" c="red">你和目标人物不能是同一个参与者。</Text>}
        </div>}
        {confirmError && <Alert color="red" role="alert">{confirmError}</Alert>}
        {!confirmed && <div className="setup-actions">
          {stage === 'import' && inRouter
            ? <Button size="sm" component={Link} to={'/projects/' + projectId + '/setup/participants'}>记录已解析，确认人物与资料</Button>
            : <Button size="sm" disabled={!canConfirm || submitting} loading={submitting} onClick={confirm} type="button">{submitting ? '正在导入……' : '确认人物并导入'}</Button>}
        </div>}
        {confirmed && <>
          <Alert color="green" role="status">导入完成，已保存 {confirmed.message_count} 条消息。后台会继续整理人物背景，现有分支不受影响。</Alert>
          {inRouter && <div className="setup-actions"><Button size="sm" component={Link} to={'/projects/' + projectId + '/world'}>查看人物背景整理状态</Button></div>}
        </>}
      </>}
    </Stack>
  )
}
