import {
  IconLayoutGrid,
  IconArrowLeft, IconArrowRight, IconCheck, IconChevronRight, IconUsers, IconSettings, IconHistory, IconMap, IconAlertCircle,
  IconActivityHeartbeat,
  IconArchive,
  IconChartBar,
  IconClock,
  IconDatabase,
  IconFolder,
  IconGitBranch,
  IconHome,
  IconMessageCircle,
  IconPackage,
  IconPhone,
  IconPlayerPlay,
  IconPlus,
  IconSend,
  IconTimeline,
  IconX,
} from '@tabler/icons-react'
import type { Icon as TablerIcon } from '@tabler/icons-react'

export type IconName =
  | 'projects'
  | 'back' | 'next' | 'check' | 'chevron' | 'people' | 'settings' | 'history' | 'map' | 'warning'
  | 'conversation'
  | 'activity'
  | 'archive'
  | 'branch'
  | 'chart'
  | 'clock'
  | 'close'
  | 'database'
  | 'folder'
  | 'home'
  | 'model'
  | 'nodes'
  | 'phone'
  | 'play'
  | 'plus'
  | 'send'
  | 'timeline'

type Props = {
  name: IconName
  size?: number
  className?: string
}

export function Icon({ name, size = 18, className }: Props) {
  const Component = icons[name]
  return <Component aria-hidden className={className} size={size} stroke={1.75} />
}

const icons: Record<IconName, TablerIcon> = {
  projects: IconLayoutGrid,
  back: IconArrowLeft, next: IconArrowRight, check: IconCheck, chevron: IconChevronRight,
  people: IconUsers, settings: IconSettings, history: IconHistory, map: IconMap, warning: IconAlertCircle,
  conversation: IconMessageCircle,
  activity: IconActivityHeartbeat,
  archive: IconArchive,
  branch: IconGitBranch,
  chart: IconChartBar,
  clock: IconClock,
  close: IconX,
  database: IconDatabase,
  folder: IconFolder,
  home: IconHome,
  model: IconPackage,
  nodes: IconTimeline,
  phone: IconPhone,
  play: IconPlayerPlay,
  plus: IconPlus,
  send: IconSend,
  timeline: IconTimeline,
}
