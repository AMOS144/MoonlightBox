import { ActionIcon, Tooltip } from '@mantine/core'
import { useLocation, useNavigate } from 'react-router-dom'
import { Icon } from './Icon'

/** 直达页面没有应用内历史时，退回所属页面，而不是离开应用。 */
function parentPage(path: string) {
  const match = path.match(/^\/projects\/([^/]+)(?:\/(.*))?$/)
  if (!match || match[1] === 'new') return '/'
  const root = `/projects/${match[1]}`
  const tail = match[2] ?? ''
  if (!tail) return '/'
  if (tail.startsWith('branches/')) return `${root}/branches`
  if (tail.startsWith('node-profiles/')) return `${root}/background`
  if (tail === 'background') return `${root}/nodes`
  if (tail === 'world/places') return `${root}/world`
  if (tail === 'setup/graph') return `${root}/setup/participants`
  return root
}

export function PageBack() {
  const location = useLocation()
  const navigate = useNavigate()
  if (location.pathname === '/') return null
  return <Tooltip label="返回" position="right" openDelay={300}>
    <ActionIcon className="page-back__button" variant="default" size="lg" radius="xl" aria-label="返回"
      onClick={() => {
        // BrowserRouter 的 idx 只记录本站路由，不用浏览器 history.length 猜测。
        if (typeof window.history.state?.idx === 'number' && window.history.state.idx > 0) navigate(-1)
        else navigate(parentPage(location.pathname), { replace: true })
      }}>
      <Icon name="back" size={18} />
    </ActionIcon>
  </Tooltip>
}
