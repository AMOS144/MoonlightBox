import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Avatar, Button, Group, Skeleton, Stack, Text, TextInput } from '@mantine/core'
import { Link } from 'react-router-dom'
import { request } from '../../api/client'
import { Icon } from '../../components/Icon'
import { PageHeader } from '../../components/PageHeader'
import { AsyncState } from '../../components/feedback/AsyncState'
import type { Project } from './types'
import './projects.css'

/** 项目名用于区分资料；同一人物的多个项目不能看成重复的人物卡片。 */
export function ProjectListPage() {
  const [filter, setFilter] = useState('')
  const projects = useQuery({ queryKey: ['projects'], queryFn: () => request<Project[]>('/api/projects') })
  const visible = projects.data?.filter(p => [p.name, p.target_name].join(' ').toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase())) ?? []
  return <section className="project-library">
    <PageHeader title="所有项目" action={<Button component={Link} leftSection={<Icon name="plus" size={16} />} to="/projects/new">创建项目</Button>} />
    <AsyncState error={projects.error} retry={() => void projects.refetch()} />
    {projects.isPending && <Stack mt="lg">{[0, 1, 2].map(i => <Skeleton key={i} h={88} radius="sm" />)}</Stack>}
    {projects.data && <>
      <Group className="library-toolbar" justify="space-between">
        <Text size="xs" c="dimmed">{projects.data.length} 个项目</Text>
        {projects.data.length > 6 && <TextInput aria-label="查找项目" placeholder="查找项目或人物" value={filter} onChange={e => setFilter(e.currentTarget.value)} />}
      </Group>
      <div className="project-directory">
        {visible.map(project => <Link className="project-directory-row" key={project.id} to={'/projects/' + project.id}>
          <Avatar radius="xl" size={44} src={project.target_avatar_asset_id ? '/api/projects/' + project.id + '/media/' + project.target_avatar_asset_id : undefined} alt="">{(project.target_name ?? project.name).slice(0, 1)}</Avatar>
          <div className="project-directory-name"><Text fw={500}>{project.name}</Text><Text size="xs" c="dimmed">{project.target_name && project.target_name !== project.name ? project.target_name : project.target_name ? '聊天资料' : '尚未确认人物'}</Text></div>
          <Text className="project-directory-date" size="xs" c="dimmed">资料更新 {new Date(project.updated_at).toLocaleDateString('zh-CN')}</Text>
          <Icon name="chevron" size={16} />
        </Link>)}
      </div>
      {!visible.length && <div className="library-empty"><Text>{filter.trim() ? '没有找到匹配的项目' : '还没有项目'}</Text><Text size="sm" c="dimmed">{filter.trim() ? '试试其他名称。' : '从右上角创建项目，再导入聊天记录。'}</Text></div>}
    </>}
  </section>
}
