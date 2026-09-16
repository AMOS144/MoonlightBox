import { useEffect, useState } from 'react'
import { Burger, Divider, Drawer, Group, NavLink as MantineNavLink, Stack, Text, ThemeIcon, useMantineTheme } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { NavLink, Outlet, useLocation, useMatch } from 'react-router-dom'
import { Icon } from '../../components/Icon'
import { ProjectTaskBar } from './ProjectTaskBar'
import { PageBack } from '../../components/PageBack'

const navigation = [
  ['概览', '', 'home', ['']],
  ['人物资料', 'setup/participants', 'chart', ['setup', 'world']],
  ['选择起点', 'nodes', 'timeline', ['nodes']],
  ['起点背景', 'background', 'chart', ['background', 'node-profiles']],
  ['我的分支', 'branches', 'conversation', ['branches']],
] as const

export function ProjectLayout() {
  // 公共外壳位于项目路由上方，显式匹配当前项目，避免把 /projects/new 当作项目。
  const projectMatch = useMatch('/projects/:projectId/*')
  const projectId = projectMatch?.params.projectId !== 'new' ? projectMatch?.params.projectId : undefined
  const location = useLocation()
  const theme = useMantineTheme()
  const narrow = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`)
  const [opened, setOpened] = useState(false)
  const branchChatMatch = useMatch('/projects/:projectId/branches/:branchId')
  const isBranchChat = Boolean(branchChatMatch && branchChatMatch.params.branchId !== 'new')
  useEffect(() => {
    setOpened(false)
  }, [location.pathname, location.search])
  const nav = <Stack gap={4}>
    {location.pathname !== '/settings' && <MantineNavLink component={NavLink} to="/" end active={location.pathname === '/'} label="所有项目" leftSection={<Icon name="projects" />} />}
    {projectId && <Divider my="sm" />}
    {projectId && navigation.map(([label, path, icon, matches]) => {
      const tail = location.pathname.split(`/projects/${projectId}/`)[1] ?? ''
      const active = matches.some(match => match === '' ? tail === '' : tail === match || tail.startsWith(`${match}/`))
      return <MantineNavLink active={active} component={NavLink} end={path === ''} key={path} label={label}
        leftSection={path === 'setup/participants' ? <Icon name="people" size={18} /> : <Icon name={icon} size={18} />} to={`/projects/${projectId}/${path}`} />
    })}
  </Stack>
  const settingsLink = <MantineNavLink component={NavLink} to="/settings" label="设置"
    leftSection={<Icon name="settings" size={18} />} active={location.pathname === '/settings'} />
  return <div className={`project-layout${isBranchChat ? ' project-layout--branch-chat' : ''}`}
    style={{ gridTemplateColumns: narrow ? 'minmax(0, 1fr)' : '212px minmax(0, 1fr)' }}>
    {!narrow && <aside className="journey-sidebar">
      <NavLink className="brand" to="/"><ThemeIcon color="moon" radius="xl" size={38} variant="light">月</ThemeIcon>
        <span>月光宝盒</span>
      </NavLink>
      <nav aria-label="项目导航">{nav}</nav>
      <nav aria-label="应用设置" style={{ marginTop: 'auto', paddingTop: 16 }}>{settingsLink}</nav>
    </aside>}
    <main className={`project-content${isBranchChat ? ' project-content--branch-chat' : ''}`}
      style={isBranchChat ? { display: 'flex', flexDirection: 'column' } : undefined}>
      {narrow && <Group className="mobile-project-header" justify="space-between" p="sm"><Burger opened={opened} onClick={() => setOpened(v => !v)} aria-label="打开项目导航" size="sm" /><Text size="sm">月光宝盒</Text></Group>}
      <Drawer opened={opened} onClose={() => setOpened(false)} title="项目导航" position="left" size="xs"><Stack style={{ minHeight: 'calc(100dvh - 100px)' }}>{nav}<div style={{ marginTop: 'auto' }}>{settingsLink}</div></Stack></Drawer>
      {isBranchChat ? <><div className="page-back page-back--chat"><PageBack /></div><Outlet /></> : <div className={`project-page${location.pathname.includes('/setup/') ? ' project-page--setup' : ''}`}>
        {location.pathname !== '/' && <div className="page-back"><PageBack /></div>}
        {projectId ? <ProjectTaskBar projectId={projectId} /> : null}
        <Outlet />
      </div>}
    </main>
  </div>
}
