import { createTheme, DEFAULT_THEME, mergeMantineTheme } from '@mantine/core'

const override = createTheme({
  primaryColor: 'moon',
  defaultRadius: 'md',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
  headings: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif' },
  colors: {
    moon: ['#faf7f0', '#f1eadb', '#e5d7b8', '#d8c38f', '#c9ad65', '#b8964c', '#98773b', '#755a32', '#594528', '#493921'],
  },
  components: {
    Button: { defaultProps: { radius: 'md' } },
    Paper: { defaultProps: { radius: 'md' } },
    Card: { defaultProps: { radius: 'md' } },
  },
})

export const theme = mergeMantineTheme(DEFAULT_THEME, override)
