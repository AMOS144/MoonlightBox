import { Tabs } from '@mantine/core'
import { Link, useLocation } from 'react-router-dom'

type Item = { label: string; to: string }

export function SectionNav({ label, items }: { label: string; items: Item[] }) {
  const location = useLocation()
  const active = items.find((item) => location.pathname.endsWith(item.to.replace('../', '/')))?.to ?? null
  return (
    <Tabs aria-label={label} mb="sm" value={active}>
      <Tabs.List>
        {items.map((item) => <Tabs.Tab key={item.to} renderRoot={(props) => <Link {...props} to={item.to} />} value={item.to}>{item.label}</Tabs.Tab>)}
      </Tabs.List>
    </Tabs>
  )
}
