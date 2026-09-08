import { Group, NavLink as MantineNavLink, Stack, Text, ThemeIcon } from '@mantine/core'
import { NavLink, Outlet, useLocation, useMatch, useParams } from 'react-router-dom'

import { Icon } from '../../components/Icon'
import { ProjectTaskBar } from './ProjectTaskBar'

const navigation = [
  ['首页', '', 'home', ['']],
  ['回忆', 'timeline', 'timeline', ['data', 'events', 'timeline']],
  ['世界', 'world', 'chart', ['world']],
  ['数字人', 'models', 'model', ['models', 'evaluation', 'training']],
  ['平行时间线', 'branches', 'branch', ['branches']],
] as const

export function ProjectLayout() {
  const { projectId } = useParams()
  const location = useLocation()
  const branchChatMatch = useMatch('/projects/:projectId/branches/:branchId')
  const isBranchChat = Boolean(
    branchChatMatch && branchChatMatch.params.branchId !== 'new',
  )

  return (
    <div className={`project-layout${isBranchChat ? ' project-layout--branch-chat' : ''}`}>
      <aside className="sidebar">
        <NavLink className="brand" to="/">
          <ThemeIcon color="moon" radius="xl" size={38} variant="light">月</ThemeIcon>
          <span>月光宝盒<Text c="dimmed" size="10px">Moonlight Box</Text></span>
        </NavLink>
        <nav aria-label="项目导航">
          <Stack gap={4}>
          {navigation.map(([label, path, icon, matches]) => {
            const tail = location.pathname.split(`/projects/${projectId}/`)[1] ?? ''
            const active = matches.some((match) => match === '' ? tail === '' : tail === match || tail.startsWith(`${match}/`))
            return (
            <MantineNavLink active={active} component={NavLink} end={path === ''} key={path} label={label} leftSection={<Icon name={icon} size={18} />} to={path} />
          )})}
          </Stack>
        </nav>
        <Group className="sidebar__back" gap={6} renderRoot={(props) => <NavLink {...props} to="/" />}><span>←</span><Text size="sm">所有项目</Text></Group>
      </aside>
      <main
        className={`project-content${isBranchChat ? ' project-content--branch-chat' : ''}`}
      >
        {projectId && !isBranchChat ? <ProjectTaskBar projectId={projectId} /> : null}
        <Outlet />
      </main>
    </div>
  )
}
