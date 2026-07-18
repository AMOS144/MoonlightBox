import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useParams } from 'react-router-dom'

import { request } from '../../api/client'
import type { BranchMessage } from './types'

export function BranchChatPage() {
  const { projectId, branchId } = useParams()
  const [content, setContent] = useState('')
  const queryClient = useQueryClient()
  const messages = useQuery({
    queryKey: ['branch-messages', branchId],
    queryFn: () =>
      request<BranchMessage[]>(
        `/api/projects/${projectId}/branches/${branchId}/messages`,
      ),
    enabled: Boolean(projectId && branchId),
  })
  const send = useMutation({
    mutationFn: () =>
      request<BranchMessage>(
        `/api/projects/${projectId}/branches/${branchId}/messages`,
        {
          method: 'POST',
          body: JSON.stringify({ content }),
        },
      ),
    onSuccess: async () => {
      setContent('')
      await queryClient.invalidateQueries({ queryKey: ['branch-messages', branchId] })
    },
  })

  return (
    <section className="branch-chat">
      <p className="eyebrow">平行时间线</p>
      <h1>从这里重新选择</h1>
      <div className="message-list">
        {messages.data?.map((message) => (
          <p className={`message message--${message.role}`} key={message.id}>
            {message.content}
          </p>
        ))}
      </div>
      <form
        className="message-form"
        onSubmit={(event) => {
          event.preventDefault()
          if (content.trim()) send.mutate()
        }}
      >
        <label htmlFor="branch-message">输入新的选择</label>
        <textarea
          id="branch-message"
          onChange={(event) => setContent(event.target.value)}
          value={content}
        />
        <button className="primary-button" disabled={send.isPending} type="submit">
          发送
        </button>
      </form>
    </section>
  )
}
