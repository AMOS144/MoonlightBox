import type {
  ProfileStatementSelection,
  StoredProfileStatementSelection,
} from './types'

export function profileSelectionKey(
  section: string,
  text: string,
  sourceMessageIds: string[],
  claimId?: string,
) {
  // 已有 v2 事实优先使用数据库 Claim ID；文本与来源仅用于旧 Profile 的只读兼容。
  return claimId
    ? `claim:${claimId}`
    : JSON.stringify([section, text, [...sourceMessageIds].sort()])
}

export function restoreProfileSelections(
  items: StoredProfileStatementSelection[],
): ProfileStatementSelection[] {
  return items.map((item) => ({
    key: item.field_path !== undefined ? `v3:${item.section}:${item.module_id ?? item.entry_id ?? ''}:${item.field_path}` : profileSelectionKey(item.section, item.text, item.source_message_ids, item.claim_id),
    entry_id: item.entry_id,
    module_id: item.module_id,
    field_path: item.field_path,
    section: item.section,
    sectionLabel: item.section_label,
    statement: {
      claim_id: item.claim_id,
      text: item.text,
      source_status: 'human_corrected',
      source_document_ids: [],
      source_message_ids: item.source_message_ids,
    },
  }))
}
