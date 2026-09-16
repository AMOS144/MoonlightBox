/** 用户提示与技术诊断分开；未知异常不直接输出堆栈、路径或服务内部字段。 */
export function userMessage(value: unknown, status?: number, code?: string): string {
  if (code === 'lightrag_content_rejected') return '抽取模型的内容审核拒绝处理该聊天片段，已停止自动重试。已完成进度和原始资料均已保留。'
  if (code === 'lightrag_provider_request_rejected') return '抽取模型拒绝了请求，请检查模型配置和受阻片段。已有进度已保留，需要处理后恢复。'
  if (code === 'lightrag_provider_authentication') return '抽取模型的凭据验证失败，请检查图谱抽取服务配置。已有进度已保留。'
  if (code === 'lightrag_index_error') return '资料整理遇到内部错误，已暂停自动恢复。已有进度已保留，请检查后恢复。'
  const raw = value instanceof Error ? value.message : typeof value === 'string' ? value : ''
  if (/authentication|api.?key|认证失败/i.test(raw)) return '模型服务未能验证凭据，请检查设置中的 API Key。'
  if (status === 401 || status === 403) return '当前操作未获授权，请检查服务访问配置。'
  if (status === 429 || /rate.?limit/i.test(raw)) return '服务暂时繁忙，请稍后再试。'
  if (/timeout|超时/i.test(raw)) return '这次处理用时较长，请稍后重试。'
  if (/Failed to fetch|NetworkError|连接失败|无法连接/i.test(raw)) return '暂时连不上服务，请检查连接后重试。'
  if (status === 404 || /项目不存在|分支不存在/.test(raw)) return '找不到这项内容，请返回项目列表重新进入。'
  if (status === 409) return '内容状态已变化，请刷新后再操作。'
  if (status === 422) return '填写的内容不完整或格式不正确，请检查后再试。'
  if (!raw || raw === '请求失败' || (status && status >= 500) || raw.length > 180 ||
    /[a-z]+_[a-z_]+|Traceback|Exception|Error:|\/home\/|\/api\/|Sidecar|Phoenix|SQLAlchemy|\bHTTP\b|\bJSON\b|\bWorker\b/i.test(raw) || !/[\u4e00-\u9fff]/.test(raw)) {
    return '暂时无法完成操作，请稍后重试。'
  }
  return raw
}

export function mediaNotice(reason: string, count: number): string {
  return reason === 'missing_strong_media_reference'
    ? `${count} 条表情消息暂未找到对应文件，不影响导入聊天文字。可以检查导出目录是否包含表情文件。`
    : `${count} 个媒体文件暂未完成关联，不影响导入聊天文字。可以检查导出目录中的媒体文件。`
}
