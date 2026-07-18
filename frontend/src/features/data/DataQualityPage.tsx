import { useParams } from 'react-router-dom'

import { ImportWizard } from './ImportWizard'

export function DataQualityPage() {
  const { projectId } = useParams()

  if (!projectId) {
    return <p role="alert">缺少项目标识。</p>
  }

  return (
    <div>
      <ImportWizard projectId={projectId} />
      <section className="quality-notes">
        <h2>数据处理原则</h2>
        <ul>
          <li>原始消息保持不变，清洗和脱敏结果单独保存。</li>
          <li>连续短回复先合并，不按单字规则直接删除。</li>
          <li>无法解析的行会单独报告，不影响其他有效消息。</li>
        </ul>
      </section>
    </div>
  )
}
