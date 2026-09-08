import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, test } from 'vitest'

import { MessageBubble } from './MessageBubble'
import type { BranchMessage, MessageType } from './types'

function message(type: MessageType, mediaAssetId: string | null): BranchMessage {
  return {
    id: 'message-1',
    branch_id: 'branch-1',
    sequence: 1,
    role: 'assistant',
    content: type === 'call' ? '视频通话 · 对方无应答' : `[${type}]`,
    type,
    media_asset_id: mediaAssetId,
    turn_id: 'turn-1',
    bubble_index: 0,
    delay_ms: 0,
    generation_status: 'completed',
    generation_metadata: {},
    created_at: '2026-07-20T00:00:00Z',
  }
}

describe('MessageBubble 媒体渲染', () => {
  test('点击图片会打开大图预览', () => {
    const { rerender } = render(
      <MessageBubble
        message={message('image', 'image-1')}
        projectId="project-1"
        showAvatar={false}
      />,
    )
    expect(screen.getByRole('img', { name: '图片' })).toHaveAttribute(
      'src',
      '/api/projects/project-1/media/image-1',
    )
    fireEvent.click(screen.getByRole('button', { name: '查看图片' }))
    expect(screen.getByRole('dialog', { name: '图片预览' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '关闭图片预览' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    rerender(
      <MessageBubble
        message={message('audio', 'audio-1')}
        projectId="project-1"
        showAvatar={false}
      />,
    )
    expect(screen.getByRole('button', { name: '播放语音' })).toBeInTheDocument()
    expect(screen.getByLabelText('语音消息')).toHaveAttribute(
      'src',
      '/api/projects/project-1/media/audio-1',
    )
    expect(screen.getByLabelText('语音消息')).not.toHaveAttribute('controls')
  })

  test('通话显示摘要，不展示协议文本', () => {
    render(
      <MessageBubble
        message={message('call', null)}
        projectId="project-1"
        showAvatar={false}
      />,
    )

    expect(screen.getByText('视频通话 · 对方无应答')).toBeInTheDocument()
    expect(screen.queryByText(/voipmsg/)).not.toBeInTheDocument()
  })

  test('引用消息显示引用卡片并隐藏微信协议标识', () => {
    render(
      <MessageBubble
        message={{
          ...message('quote', null),
          content: JSON.stringify({
            text: '正是这样',
            quoted_content: '仍在加班中么',
          }),
        }}
        projectId="project-1"
        showAvatar={false}
      />,
    )

    expect(screen.getByText('正是这样')).toBeInTheDocument()
    expect(screen.getByText('仍在加班中么')).toBeInTheDocument()
    expect(screen.queryByText(/wxid_/)).not.toBeInTheDocument()
  })

  test('系统消息不显示成聊天气泡', () => {
    const { container } = render(
      <MessageBubble
        message={{ ...message('system', null), content: '撤回了一条消息' }}
        projectId="project-1"
        showAvatar
      />,
    )

    expect(screen.getByText('撤回了一条消息')).toBeInTheDocument()
    expect(container.querySelector('.chat-avatar')).not.toBeInTheDocument()
    expect(container.querySelector('.chat-system-message')).toBeInTheDocument()
  })

  test('反应显示为轻量系统反馈，已撤回内容不再显示', () => {
    const { container, rerender } = render(
      <MessageBubble
        message={{
          ...message('reaction', null),
          content: JSON.stringify({ reaction: '拍了拍' }),
        }}
        projectId="project-1"
        showAvatar
      />,
    )
    expect(screen.getByText('对方拍了拍你')).toBeInTheDocument()
    expect(container.querySelector('.chat-avatar')).not.toBeInTheDocument()

    rerender(
      <MessageBubble
        message={{
          ...message('text', null),
          content: '这句已撤回',
          generation_metadata: { retracted: true },
        }}
        projectId="project-1"
        showAvatar
      />,
    )
    expect(screen.queryByText('这句已撤回')).not.toBeInTheDocument()
  })

  test('兼容没有生成元数据的历史消息', () => {
    const historical = message('text', null)
    delete (historical as Partial<BranchMessage>).generation_metadata

    render(
      <MessageBubble
        message={historical}
        projectId="project-1"
        showAvatar={false}
      />,
    )

    expect(screen.getByText('[text]')).toBeInTheDocument()
  })
})
