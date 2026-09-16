import { useCallback, useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { request } from '../../api/client'
import type { BranchMessage } from './types'

type Pending = { message: BranchMessage; status: 'sending' | 'failed' | 'sent' }
type Receipt = { message: BranchMessage }
const EMPTY_PENDING: Pending[] = []

/** 每个分支独立保存待发送消息；迟到回执只更新原分支，不覆盖新草稿。 */
export function useBranchOutbox(projectId: string, branchId: string, persisted: BranchMessage[]) {
  const client = useQueryClient()
  const key = useMemo(() => ['branch-outbox', projectId, branchId], [projectId, branchId])
  const storageKey = `branch:${projectId}:${branchId}:outbox`
  const query = useQuery<Pending[]>({
    queryKey: key, enabled: false, gcTime: Infinity,
    initialData: () => {
      try {
        const data = JSON.parse(sessionStorage.getItem(storageKey) ?? '[]') as Pending[]
        return data.filter(item => item.message?.branch_id === branchId && item.message.client_message_id)
          .map(item => ({ ...item, status: item.status === 'sent' ? 'sent' : 'failed' }))
      } catch { return [] }
    },
  })
  const update = useCallback((change: (items: Pending[]) => Pending[]) => {
    client.setQueryData<Pending[]>(key, current => {
      const value = change(current ?? [])
      try { sessionStorage.setItem(storageKey, JSON.stringify(value)) } catch { /* 容量不足仍保留内存数据。 */ }
      return value
    })
  }, [client, key, storageKey])
  const delivered = useMemo(() => new Set(persisted.map(message => message.client_message_id)), [persisted])
  const pending = query.data ?? EMPTY_PENDING
  useEffect(() => {
    if (pending.some(item => delivered.has(item.message.client_message_id))) {
      update(items => items.filter(item => !delivered.has(item.message.client_message_id)))
    }
  }, [delivered, pending, update]) // 数据落盘已由 GET 确认，清理待发送副本。

  const send = useMutation({
    mutationFn: async (message: BranchMessage) => {
      const receipt = await request<Receipt>(`/api/projects/${projectId}/branches/${branchId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content: message.content, client_message_id: message.client_message_id }),
      })
      if (receipt?.message?.client_message_id !== message.client_message_id
          || receipt.message.branch_id !== branchId) throw new Error('消息回执不完整，请重试确认')
      return receipt.message
    },
    onSuccess: message => {
      update(items => items.map(item => item.message.client_message_id === message.client_message_id
        ? { message, status: 'sent' } : item))
      void client.invalidateQueries({ queryKey: ['branch-messages', branchId] })
    },
    onError: (_error, message) => {
      update(items => items.map(item => item.message.client_message_id === message.client_message_id
        ? { ...item, status: 'failed' } : item))
    },
  })
  function enqueue(message: BranchMessage) {
    // 同一消息在途不能再次提交；失败重试仍沿用原 client_message_id。
    const existing = client.getQueryData<Pending[]>(key)?.find(
      item => item.message.client_message_id === message.client_message_id,
    )
    if (existing && existing.status !== 'failed') return
    update(items => existing
      ? items.map(item => item.message.client_message_id === message.client_message_id
        ? { message, status: 'sending' } : item)
      : [...items, { message, status: 'sending' }])
    send.mutate(message)
  }
  return { pending: pending.filter(item => !delivered.has(item.message.client_message_id)), enqueue }
}
