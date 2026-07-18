import { NavLink, Outlet } from 'react-router-dom'

const navigation = [
  ['概览', ''],
  ['数据', 'data'],
  ['节点', 'events'],
  ['模型', 'models'],
  ['时间轴', 'timeline'],
  ['分支', 'branches'],
  ['评测', 'evaluation'],
] as const

export function ProjectLayout() {
  return (
    <div className="project-layout">
      <aside className="sidebar">
        <NavLink className="brand" to="/">
          月光宝盒
        </NavLink>
        <nav>
          {navigation.map(([label, path]) => (
            <NavLink end={path === ''} key={path} to={path}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="project-content">
        <Outlet />
      </main>
    </div>
  )
}
