import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import type { BranchMessage } from './types'

type Props = {
  message: BranchMessage
  projectId: string
  showAvatar: boolean
  avatarAssetId?: string | null
  avatarName?: string
}

export function MessageBubble({
  message,
  projectId,
  showAvatar,
  avatarAssetId,
  avatarName,
}: Props) {
  const isUser = message.role === 'user'
  if (message.generation_metadata?.retracted === true) {
    return null
  }
  if (message.type === 'system') {
    return <div className="chat-system-message">{message.content}</div>
  }
  if (message.type === 'reaction') {
    return <div className="chat-system-message">{reactionLabel(message.content)}</div>
  }
  const mediaUrl = message.media_asset_id
    ? `/api/projects/${projectId}/media/${message.media_asset_id}`
    : null
  const avatar = showAvatar ? (
    avatarAssetId ? (
      <img
        alt={`${avatarName ?? '联系人'}头像`}
        className="chat-avatar"
        src={`/api/projects/${projectId}/media/${avatarAssetId}`}
      />
    ) : (
      <div className={`chat-avatar${isUser ? ' chat-avatar--self' : ''}`}>
        {(avatarName ?? (isUser ? '我' : '小')).slice(0, 1)}
      </div>
    )
  ) : (
    <div className="chat-avatar chat-avatar--empty" />
  )
  return (
    <div className={`chat-row chat-row--${message.role}`}>
      {!isUser ? avatar : null}
      <MessageContent
        content={message.content}
        mediaUrl={mediaUrl}
        role={message.role}
        type={message.type}
      />
      {isUser ? avatar : null}
    </div>
  )
}

function reactionLabel(content: string): string {
  try {
    const payload = JSON.parse(content) as { reaction?: unknown }
    if (payload.reaction === '拍了拍') return '对方拍了拍你'
  } catch {
    // 保留异常历史记录的可读降级。
  }
  return '对方回应了这条消息'
}

function MessageContent({
  content,
  mediaUrl,
  role,
  type,
}: {
  content: string
  mediaUrl: string | null
  role: BranchMessage['role']
  type: BranchMessage['type']
}) {
  if (type === 'sticker' && mediaUrl) {
    return <img alt="表情" className="chat-sticker" src={mediaUrl} />
  }
  if (type === 'image' && mediaUrl) {
    return <ImageMessage mediaUrl={mediaUrl} />
  }
  if (type === 'video' && mediaUrl) {
    return (
      <video
        aria-label="视频消息"
        className="chat-video"
        controls
        preload="metadata"
        src={mediaUrl}
      />
    )
  }
  if (type === 'video_thumbnail' && mediaUrl) {
    return (
      <div className="chat-video-thumbnail">
        <img alt="视频封面" className="chat-video" loading="lazy" src={mediaUrl} />
        <span aria-hidden="true">▶</span>
      </div>
    )
  }
  if (type === 'audio' && mediaUrl) {
    return <VoiceMessage mediaUrl={mediaUrl} role={role} />
  }
  if (type === 'quote') {
    return <QuoteMessage content={content} role={role} />
  }
  const label =
    type === 'image'
      ? '图片暂未下载'
      : type === 'video' || type === 'video_thumbnail'
        ? '视频暂未下载'
        : type === 'audio'
          ? '语音暂不可播放'
          : content
  return (
    <div
      className={`chat-bubble chat-bubble--${role}${type === 'call' ? ' chat-bubble--call' : ''}${type !== 'text' && type !== 'unknown' ? ' chat-bubble--media-placeholder' : ''}`}
    >
      {type === 'call' ? <span aria-hidden="true">☎</span> : null}
      {label}
    </div>
  )
}

function ImageMessage({ mediaUrl }: { mediaUrl: string }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        aria-label="查看图片"
        className="chat-image-button"
        onClick={() => setOpen(true)}
        type="button"
      >
        <img alt="图片" className="chat-image" loading="lazy" src={mediaUrl} />
      </button>
      {open
        ? createPortal(
            <div
              aria-label="图片预览"
              aria-modal="true"
              className="chat-image-preview"
              onClick={() => setOpen(false)}
              role="dialog"
            >
              <button
                aria-label="关闭图片预览"
                className="chat-image-preview__close"
                onClick={() => setOpen(false)}
                type="button"
              >
                ×
              </button>
              <img
                alt="图片大图"
                onClick={(event) => event.stopPropagation()}
                src={mediaUrl}
              />
            </div>,
            document.body,
          )
        : null}
    </>
  )
}

function VoiceMessage({
  mediaUrl,
  role,
}: {
  mediaUrl: string
  role: BranchMessage['role']
}) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [duration, setDuration] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const toggle = () => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
      return
    }
    void audio.play().then(
      () => setPlaying(true),
      () => setPlaying(false),
    )
  }
  return (
    <div className={`chat-voice chat-voice--${role}`}>
      <button
        aria-label={playing ? '暂停语音' : '播放语音'}
        className={`chat-voice__bubble${playing ? ' chat-voice__bubble--playing' : ''}`}
        onClick={toggle}
        type="button"
      >
        <span aria-hidden="true" className="chat-voice__waves">
          <i />
          <i />
          <i />
        </span>
      </button>
      <span className="chat-voice__duration">
        {duration === null ? '' : `${duration}″`}
      </span>
      <audio
        aria-label="语音消息"
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(event) => {
          const seconds = event.currentTarget.duration
          setDuration(Number.isFinite(seconds) ? Math.max(1, Math.round(seconds)) : null)
        }}
        preload="metadata"
        ref={audioRef}
        src={mediaUrl}
      />
    </div>
  )
}

function QuoteMessage({
  content,
  role,
}: {
  content: string
  role: BranchMessage['role']
}) {
  let text = content
  let quotedContent = "[消息]"
  try {
    const value = JSON.parse(content) as {
      text?: unknown
      quoted_content?: unknown
    }
    if (typeof value.text === 'string') text = value.text
    if (typeof value.quoted_content === 'string') {
      quotedContent = value.quoted_content
    }
  } catch {
    // 历史数据格式异常时仍保留可读正文。
  }
  return (
    <div className={`chat-bubble chat-bubble--${role} chat-bubble--quote`}>
      <blockquote>{quotedContent}</blockquote>
      <span>{text}</span>
    </div>
  )
}
