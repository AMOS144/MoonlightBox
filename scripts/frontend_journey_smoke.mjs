// 只读浏览器冒烟：连接独立的本地 Chrome 调试实例，不发送聊天或启动 Agent。
// 用法：node scripts/frontend_journey_smoke.mjs <project_id> [debug_port]
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const [project, debugPort = '9224'] = process.argv.slice(2)
if (!project) throw new Error('需要项目 ID')
const host = `http://127.0.0.1:${debugPort}`
const target = await (await fetch(`${host}/json/new?about:blank`, { method: 'PUT' })).json()
const socket = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject })
let nextId = 0
const calls = new Map()
const exceptions = []
socket.onmessage = event => {
  const value = JSON.parse(event.data)
  if (value.id) {
    const call = calls.get(value.id)
    calls.delete(value.id)
    if (value.error) call?.reject(new Error(value.error.message))
    else call?.resolve(value.result)
  } else if (value.method === 'Runtime.exceptionThrown') {
    exceptions.push(value.params.exceptionDetails.text)
  }
}
function send(method, params = {}) {
  const id = ++nextId
  return new Promise((resolve, reject) => { calls.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })) })
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
  return result.result.value
}
async function until(expression) {
  const end = Date.now() + 15000
  while (Date.now() < end) {
    if (await evaluate(`Boolean(${expression})`)) return
    await new Promise(resolve => setTimeout(resolve, 250))
  }
  throw new Error(`页面未就绪: ${expression}`)
}
await send('Page.enable')
await send('Runtime.enable')
const screenshots = await mkdtemp(join(tmpdir(), 'moonlight-journey-ui-'))
const root = `/projects/${project}`
const journey = await (await fetch(`http://127.0.0.1:5175/api${root}/journey`)).json()
const cases = [
  ['/', 'h1', '项目列表', true],
  ['/projects/new', 'h1', '创建项目', true],
  ['', 'h1', '概览'], ['/setup/participants', 'h1', '资料'],
  ['/world?view=published', 'h1', '背景'], ['/nodes', 'h2', '起点'],
  ['/branches', 'h1', '分支'], ['/events', 'h2', '旧链接'],
  ['/advanced', 'h1', '高级设置'], ['/setup/import', 'h1', '导入'],
  ['/branches/new', 'h1', '创建分支'], ['/models', 'h1', '模型'],
  ['/evaluation', 'h1', '评估'], ['/world/places', 'h1', '地图'],
  ['/events/legacy', 'h1', '历史资料'],
]
if (journey.branches?.[0]) {
  cases.push([`/branches/${journey.branches[0].id}`, '.branch-chat__header', '已有聊天'])
  cases.push([`/branches/${journey.branches[0].id}/preparing`, 'h2', '准备预览'])
}
try {
  for (const width of [1440, 768, 390]) {
    await send('Emulation.setDeviceMetricsOverride', { width, height: 1000, deviceScaleFactor: 1, mobile: false })
    for (const [path, selector, name, absolute] of cases) {
      await send('Page.navigate', { url: `http://127.0.0.1:5175${absolute ? path : root + path}` })
      await until(`document.querySelector(${JSON.stringify(selector)}) && !document.body.innerText.includes('正在读取人物背景')`)
      await new Promise(resolve => setTimeout(resolve, 700))
      const state = await evaluate(`({ overflow: document.documentElement.scrollWidth > innerWidth + 2, title: document.querySelector(${JSON.stringify(selector)}).textContent, path: location.pathname, error: document.body.innerText.includes('流程状态读取失败') })`)
      if (state.error || state.overflow) throw new Error(`${width} ${name}: ${JSON.stringify(state)}`)
      if (path === '/events' && !state.path.endsWith('/nodes')) throw new Error('旧链接没有回归新起点')
      if (path === '/world?view=published') {
        await until(`document.querySelector('.profile-reading-layout')`)
        await evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.textContent === '修改背景')?.click()`)
        await until(`Array.from(document.querySelectorAll('button')).some(b => b.textContent === '返回阅读')`)
        if (!await evaluate(`document.querySelector('[aria-label^="选择"]') !== null`)) throw new Error('修改模式缺少字段选择')
        // 仅切换界面，不创建修订或提交业务修改。
        await evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.getAttribute('aria-label') === '收起面板（不会取消任务）')?.click()`)
        await evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.textContent === '返回阅读')?.click()`)
        await until(`!Array.from(document.querySelectorAll('button')).some(b => b.textContent === '返回阅读')`)
      }
      if (name === '已有聊天') {
        await evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.textContent === '资料')?.click()`)
        await until(`document.querySelector('[role="dialog"]')?.textContent.includes('当天日程')`)
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 })
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 })
        await until(`!document.querySelector('[role="dialog"]')`)
      }
      if (path === '' && width === 1440) {
        await send('DOM.enable'); await send('CSS.enable')
        const { root: dom } = await send('DOM.getDocument')
        const { nodeId } = await send('DOM.querySelector', { nodeId: dom.nodeId, selector: '.person-home h2' })
        console.log(JSON.stringify({ actualFonts: (await send('CSS.getPlatformFontsForNode', { nodeId })).fonts }))
      }
      if (width === 390) {
        await evaluate(`document.querySelector('[aria-label="打开项目导航"]').click()`)
        await until(`document.querySelector('[role="dialog"]')`)
        const nav = await evaluate(`document.querySelector('[role="dialog"]').innerText`)
        if (!nav.includes('所有项目') || (!absolute && (!nav.includes('人物资料') || !nav.includes('我的分支')))) throw new Error('窄屏导航缺失')
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 })
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 })
      }
      const shot = await send('Page.captureScreenshot', { format: 'png' })
      await writeFile(join(screenshots, `${width}-${name}.png`), Buffer.from(shot.data, 'base64'))
      console.log(JSON.stringify({ width, name, ...state }))
    }
  }
  if (exceptions.length) throw new Error(`浏览器异常: ${exceptions.join(', ')}`)
  console.log(`screenshots: ${screenshots}`)
} finally {
  socket.close()
  await fetch(`${host}/json/close/${target.id}`)
}
