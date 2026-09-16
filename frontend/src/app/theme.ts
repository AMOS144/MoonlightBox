import { createTheme, DEFAULT_THEME, mergeMantineTheme } from '@mantine/core'
import type { AlertProps, ButtonProps, CSSVariablesResolver, MantineTheme } from '@mantine/core'

const override = createTheme({
  primaryColor: 'moon',
  autoContrast: true,
  primaryShade: { light: 6, dark: 4 },
  defaultRadius: 'md',
  fontSizes: { xs: '0.75rem', sm: '0.8125rem', md: '0.875rem', lg: '1rem', xl: '1.125rem' },
  fontFamily: '"PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif',
  lineHeights: { md: '1.65' },
  headings: { fontFamily: 'inherit', fontWeight: '600',
    sizes: { h1: { fontSize: '1.375rem', lineHeight: '1.4' }, h2: { fontSize: '1.125rem', lineHeight: '1.4' }, h3: { fontSize: '1rem', lineHeight: '1.4' }, h4: { fontSize: '0.9375rem', lineHeight: '1.5' }, h5: { fontSize: '0.875rem', lineHeight: '1.5' } } },
  spacing: { xs: '0.375rem', sm: '0.5rem', md: '0.75rem', lg: '1rem', xl: '1.25rem' },
  colors: {
    moon: ['#faf7f0', '#f1eadb', '#e5d7b8', '#d8c38f', '#c9ad65', '#b8964c', '#98773b', '#755a32', '#594528', '#493921'],
  },
  components: {
    Button: {
      defaultProps: { radius: 'sm', size: 'compact-sm' },
      // 主色实心按钮使用深色字；不覆盖危险操作、浅色按钮和禁用态。
      styles: (_theme: MantineTheme, props: ButtonProps) => ({ root: {
        fontWeight: 500,
        ...(!props.disabled && (!props.variant || props.variant === 'filled') && (!props.color || props.color === 'moon')
          ? { color: 'var(--accent-ink)' } : {}),
      } }),
    },
    Paper: { defaultProps: { radius: 'md' } },
    Card: { defaultProps: { radius: 'md' }, styles: { root: { background: 'var(--surface)' } } },
    NavLink: { styles: { root: { borderRadius: 'var(--mantine-radius-sm)', padding: '9px 10px' }, label: { fontWeight: 500, fontSize: 'var(--mantine-font-size-sm)' } } },
    Badge: { styles: { root: { textTransform: 'none', fontWeight: 500, letterSpacing: 0 } } },
    Alert: {
      defaultProps: { radius: 'sm', variant: 'light' },
      styles: (_theme: MantineTheme, props: AlertProps) => ({
        root: { background: 'var(--surface)', border: '1px solid var(--line)',
          borderInlineStart: `2px solid ${props.color === 'red' ? 'var(--notice-danger)' : 'var(--accent)'}`,
          padding: '12px 14px', color: 'var(--text)' },
        title: { color: 'var(--text)', fontSize: 'var(--mantine-font-size-sm)', fontWeight: 500 },
        message: { color: 'var(--muted)', fontSize: 'var(--mantine-font-size-sm)', lineHeight: 1.65 },
        icon: { color: props.color === 'red' ? 'var(--notice-danger)' : 'var(--accent)' },
      }),
    },
    TextInput: { defaultProps: { size: 'sm' } },
    Textarea: { defaultProps: { size: 'sm' } },
    Select: { defaultProps: { size: 'sm' } },
    Modal: { defaultProps: { centered: true, radius: 'md' } },
  },
})

export const theme = mergeMantineTheme(DEFAULT_THEME, override)

/** 老聊天样式与 Mantine 从同一个主题入口取色，避免全局 CSS 另养一套设计变量。 */
export const cssVariablesResolver: CSSVariablesResolver = (current) => ({
  variables: {
    '--background': '#11100f', '--sidebar': '#151311', '--surface': '#191716',
    '--surface-raised': '#211e1c', '--surface-soft': '#28231f', '--line': '#342f2b',
    '--line-strong': '#4a423a', '--text': '#f2eee8', '--muted': '#aaa19a', '--faint': '#7e766f',
    '--accent': current.colors.moon[4], '--accent-strong': current.colors.moon[3], '--accent-ink': '#221b0f',
    '--memory': current.colors.violet[3], '--success': current.colors.green[4], '--danger': current.colors.red[4],
    '--notice-danger': '#b78479',
    '--radius-sm': current.radius.sm, '--radius-md': current.radius.md, '--radius-lg': current.radius.lg,
    '--shadow': current.shadows.md,
  },
  dark: { '--mantine-color-body': '#11100f', '--mantine-color-text': '#f2eee8', '--mantine-color-dimmed': '#aaa19a', '--mantine-color-default': '#191716', '--mantine-color-default-border': '#342f2b', '--mantine-color-default-hover': '#28231f' },
  light: {},
})
