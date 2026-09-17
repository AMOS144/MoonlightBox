import { Accordion, ActionIcon, Badge, Button, Card, Group, Stack, Text, Title, Tooltip } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import type { ProfileStatementSelection } from './types'

type Obj = Record<string, unknown>
type Catalog = {
  sections: { key: string; label: string; fields: Record<string, string> }[]
  labels: Record<string, string>
  expression_fields?: Record<string, string>
  model_groups: Record<string, Record<string, string[]>>
  modules: Record<string, { label: string; groups: Record<string, string[]>; group_labels: Record<string, string> }>
}
const obj = (v: unknown): Obj => v && typeof v === 'object' && !Array.isArray(v) ? v as Obj : {}
const list = (v: unknown): unknown[] => Array.isArray(v) ? v : []
const text = (v: unknown) => typeof v === 'string' ? v : ''
const states: Record<string, string> = { described: '已生成', unknown: '尚不了解', not_applicable: '当前不适用', current: '当前', planned: '计划中', past: '过去', paused: '暂停' }
const values: Record<string, string> = { low: '偏低', moderate: '中等', high: '偏高', mixed: '因情境而异', unknown: '尚不了解', supported: '得到满足', frustrated: '受到限制' }
const enumColors: Record<string, string> = { low: 'gray', moderate: 'blue', high: 'orange', mixed: 'violet', unknown: 'gray', supported: 'teal', frustrated: 'red' }

export function V3ProfileGrid({ projectId, profile, locked, selectedKeys, onToggle, onToggleMany, editing = false }: {
  projectId: string; profile: Obj; locked: boolean; selectedKeys: Set<string>;
  editing?: boolean;
  onToggle: (item: ProfileStatementSelection) => void
  onToggleMany?: (items: ProfileStatementSelection[], select: boolean) => void
}) {
  const catalog = useQuery({ queryKey: ['profile-v3-catalog'], queryFn: () => request<Catalog>(`/api/projects/${projectId}/world-agent/profile-v3/catalog`), staleTime: Infinity })
  if (!catalog.data) return <Text>{catalog.isError ? '画像字段目录加载失败' : '正在加载画像…'}</Text>
  const config = catalog.data
  const label = (key: string) => config.labels[key] ?? key

  function makeSelection(section: string, name: string, item: Obj, path: string, moduleId?: string): ProfileStatementSelection {
    const id = text(item.id)
    const key = `v3:${section}:${moduleId ?? id}:${path}`
    const body = text(item.description) || text(item.summary) || text(item.value)
    return {
      key, section, sectionLabel: name, label: name, entry_id: id || undefined, module_id: moduleId, field_path: path,
      statement: { text: body || name, source_status: item.basis === 'user_corrected' ? 'human_corrected' : 'inferred', source_document_ids: [], source_message_ids: list(item.reference_message_ids).map(String) },
    }
  }

  function selectable(section: string, name: string, item: Obj, path: string, moduleId?: string) {
    const sel = makeSelection(section, name, item, path, moduleId)
    const key = sel.key
    const rawValue = text(item.value)
    const enumLabel = values[rawValue]
    const isEnum = Boolean(rawValue && enumLabel)
    const description = text(item.description) || text(item.summary)
    // 短字段（一行说得清的属性）收成"名称: 值"的行内形式，不再另起正文段落。
    const compact = !isEnum && !rawValue && Boolean(description) && description.length <= 80 && !description.includes('\n')
      && !list(item.patterns).length && !text(item.context) && !text(item.roleplay_guidance)
    const chosen = selectedKeys.has(key)
    const selection = (
      <Tooltip label={chosen ? '取消选择' : '选择'}>
        <ActionIcon variant="subtle" color={chosen ? 'teal' : 'gray'} size="sm" radius="xl"
          disabled={locked} aria-label={`${chosen ? '取消选择' : '选择'}${name}`} onClick={() => onToggle(sel)}>
          <Icon name={chosen ? 'selected' : 'select'} size={16} />
        </ActionIcon>
      </Tooltip>
    )
    return <Stack key={key} gap={4} mb="sm">
      <Group justify="space-between" gap="xs" wrap="nowrap">
        <Group gap={8} wrap="nowrap" style={{ minWidth: 0 }}>
          <Text fw={500} size="sm" style={{ flexShrink: 0 }}>{name}</Text>
          {compact ? <Text size="sm" truncate>{description}</Text> : null}
        </Group>
        <Group gap={4} wrap="nowrap">
          {item.status && item.status !== 'described' ? <Badge size="xs" variant="light">{states[text(item.status)] ?? text(item.status)}</Badge> : null}
          {item.basis === 'user_corrected' ? <Text size="xs" c="teal">已纠正</Text> : null}
          {editing && selection}
        </Group>
      </Group>
      {isEnum ? <div><Badge variant="light" color={enumColors[rawValue] ?? 'gray'}>{enumLabel}</Badge></div> : null}
      {!isEnum && rawValue && rawValue !== description ? <Text size="sm" fw={600}>{rawValue}</Text> : null}
      {!compact && (description || !rawValue) ? <Text size={isEnum ? 'xs' : 'sm'} c={isEnum ? 'dimmed' : undefined} style={{ whiteSpace: 'pre-wrap' }}>{description || '尚未形成判断'}</Text> : null}
      {list(item.patterns).map((raw, index) => { const pattern = obj(raw); return <Stack key={index} gap={2} pl="sm">
        <Text size="sm" fw={500}>{text(pattern.form)}</Text>
        <Text size="xs">怎么用、何时用：{text(pattern.use_when) || '尚未说明'}</Text>
        {text(pattern.avoid_when) ? <Text size="xs" c="dimmed">何时不用：{text(pattern.avoid_when)}</Text> : null}
        {list(pattern.reference_message_ids).length ? <Text size="xs" c="dimmed">参考片段：{list(pattern.reference_message_ids).join('、')}</Text> : null}
      </Stack> })}
      {text(item.context) ? <Text size="xs" c="dimmed">适用情境：{text(item.context)}</Text> : null}
      {text(item.roleplay_guidance) ? <Text size="sm">相处建议：{text(item.roleplay_guidance)}</Text> : null}
      {text(item.reasoning_summary) || list(item.uncertainties).length ? <details><summary>理解说明与保留意见</summary>
        {text(item.reasoning_summary) ? <Text size="sm">{text(item.reasoning_summary)}</Text> : null}
        {list(item.uncertainties).map((value, index) => <Text key={index} size="xs" c="dimmed">{text(value)}</Text>)}
      </details> : null}
      {list(item.reference_message_ids).length ? <details><summary>参考片段标识</summary><Text size="xs">{list(item.reference_message_ids).join('、')}</Text></details> : null}
    </Stack>
  }

  function model(section: string, key: string, value: unknown, title?: string) {
    const data = obj(value)
    const dimensions = list(data.dimensions).map(obj)
    const grouped = config.model_groups?.[key]
    const dimension = (name: string) => {
      const item = dimensions.find(entry => entry.dimension_id === name)
      return item ? selectable(section, label(name), item, `${key}.dimensions.${name}`) : null
    }
    const modelTitle = title ?? label(key)
    return <Accordion key={key} variant="separated" chevronPosition="left">
      <Accordion.Item value={key}>
        <Accordion.Control>{modelTitle}</Accordion.Control>
        <Accordion.Panel>
          {selectable(section, `${modelTitle}整体理解`, { description: data.summary, basis: 'inferred' }, key)}
          {list(data.candidates).length ? <Text size="sm" c="dimmed">候选：{list(data.candidates).join(' / ')}（AI 推断）</Text> : null}
          {grouped ? Object.entries(grouped).map(([domain, facets]) => <div key={domain}>
            {dimension(domain)}
            <Accordion variant="contained" chevronPosition="left">
              <Accordion.Item value={domain}>
                <Accordion.Control>{label(domain)}的三个侧面</Accordion.Control>
                <Accordion.Panel>{facets.map(dimension)}</Accordion.Panel>
              </Accordion.Item>
            </Accordion>
          </div>) : dimensions.map(item => dimension(text(item.dimension_id)))}
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  }

  function moduleCard(raw: unknown) {
    const m = obj(raw), kind = text(m.kind), moduleId = text(m.id), spec = config.modules[kind]
    if (!spec) return null
    const groups = obj(m.details)
    return <Card withBorder p="md" key={moduleId}>
      <Accordion multiple defaultValue={m.status !== 'past' ? [moduleId] : []} chevronPosition="left">
        <Accordion.Item value={moduleId}>
          <Accordion.Control>{text(m.title) || spec.label} · {states[text(m.status)] ?? '尚不了解'}</Accordion.Control>
          <Accordion.Panel>
            {selectable('life_context', text(m.title) || spec.label, m, '', moduleId)}
            {Object.entries(spec.groups).map(([group, fields]) => {
              const content = obj(groups[group])
              const known = fields.filter(key => obj(content[key]).status === 'described')
              const unknown = fields.filter(key => obj(content[key]).status !== 'described')
              const groupLabel = spec.group_labels?.[group] ?? label(group)
              return <Accordion key={group} multiple defaultValue={known.length > 0 ? [group] : []} variant="contained" chevronPosition="left">
                <Accordion.Item value={group}>
                  <Accordion.Control>{groupLabel}</Accordion.Control>
                  <Accordion.Panel>
                    {selectable('life_context', `${groupLabel}整组`, { description: known.map(key => text(obj(content[key]).description)).filter(Boolean).join('；') }, `details.${group}`, moduleId)}
                    {known.map(key => selectable('life_context', label(key), obj(content[key]), `details.${group}.${key}`, moduleId))}
                    {unknown.length ? <Accordion variant="contained" chevronPosition="left">
                      <Accordion.Item value="unknown">
                        <Accordion.Control>尚不了解或不适用（{unknown.length}）</Accordion.Control>
                        <Accordion.Panel>{unknown.map(key => selectable('life_context', label(key), obj(content[key]), `details.${group}.${key}`, moduleId))}</Accordion.Panel>
                      </Accordion.Item>
                    </Accordion> : null}
                  </Accordion.Panel>
                </Accordion.Item>
              </Accordion>
            })}
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Card>
  }

  return <div className="profile-reading-layout">
    <nav className="profile-section-nav" aria-label="人物背景栏目">{config.sections.map(section => <a key={section.key} href={`#profile-${section.key}`}>{section.label}</a>)}</nav>
    <Stack className="profile-reading-content"><Card withBorder p="lg"><Title order={3}>整体画像</Title><Text style={{ whiteSpace: 'pre-wrap' }}>{text(profile.overview) || '整体画像尚未生成'}</Text></Card>
    {config.sections.map(section => {
      const data = obj(profile[section.key])
      const sectionSels = [
        makeSelection(section.key, `${section.label}综述`, { description: data.summary }, 'summary'),
        ...Object.entries(section.fields).map(([key, title]) => makeSelection(section.key, title, obj(data[key]), key)),
      ]
      const allSelected = sectionSels.every(sel => selectedKeys.has(sel.key))
      return <Card id={`profile-${section.key}`} key={section.key} withBorder p="lg"><Stack>
        <Group justify="space-between" wrap="nowrap">
          <Title order={3}>{section.label}</Title>
          {editing && onToggleMany ? <Button size="compact-xs" variant={allSelected ? 'filled' : 'subtle'} disabled={locked}
            aria-label={`${allSelected ? '取消选择' : '选择'}${section.label}整栏`}
            onClick={() => onToggleMany(sectionSels, !allSelected)}>{allSelected ? '取消整栏' : '选择整栏'}</Button> : null}
        </Group>
        {selectable(section.key, `${section.label}综述`, { description: data.summary }, 'summary')}
        {Object.entries(section.fields).map(([key, title]) => selectable(section.key, title, obj(data[key]), key))}
        {section.key === 'relationship_with_user' ? <Accordion chevronPosition="left" variant="separated">
          <Accordion.Item value="expression">
            <Accordion.Control>人物表达资料</Accordion.Control>
            <Accordion.Panel>
              {data.expression_profile ? Object.entries(config.expression_fields ?? {}).map(([key, title]) => selectable(section.key, title, obj(obj(data.expression_profile)[key]), `expression_profile.${key}`)) : <Text size="sm" c="dimmed">此版本尚未编译表达资料，需重新调查本栏目并审核发布。</Text>}
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion> : null}
        {['big_five_bfi2', 'mbti', 'motivation', 'interpersonal_style'].filter(key => key in data).map(key => model(section.key, key, data[key]))}
        {list(data.interpersonal_styles).map(raw => { const style = obj(raw); return <div key={text(style.id)}><Title order={5}>{text(style.relationship_scope)}</Title>{model(section.key, `interpersonal_styles.${text(style.id)}.profile`, style.profile, '相处方式（IPC）')}</div> })}
        {list(data.context_modules).map(moduleCard)}
      </Stack></Card>
    })}</Stack>
  </div>
}
