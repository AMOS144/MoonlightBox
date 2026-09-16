// 创建流程的隔离浏览器检查：拦截全部 API 请求，绝不向真实后端发送写操作。
// 需要独立 Chrome 调试实例：node scripts/frontend_setup_smoke.mjs [9224]
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const host = `http://127.0.0.1:${process.argv[2] ?? '9224'}`
const target = await (await fetch(`${host}/json/new?about:blank`, { method: 'PUT' })).json()
const socket = new WebSocket(target.webSocketDebuggerUrl)
await new Promise(resolve => { socket.onopen = resolve })
let next = 0, saved = false, failCreate = true, failScan = true, failSave = true
const pending = new Map(), errors = [], writes = []
const columnWidths = new Map()
const directory = await mkdtemp(join(tmpdir(), 'moonlight-setup-mock-'))
const project = { id: 'ui-preview', name: '大学时期的聊天', status: 'created' }
const preview = { id: 'preview-ui', message_count: 1240, participants: ['小林', '小周'], time_range: ['2026-05-01T10:00:00+08:00', '2026-05-08T21:30:00+08:00'], media_file_count: 8, linked_sticker_count: 3, unlinked_sticker_count: 0, deduplicated_asset_count: 8, avatar_status: { 小周: false }, errors: [], failure_reasons: {} }
function send(method, params = {}) {
  const id = ++next
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })) })
}
async function respond({ requestId, request }) {
  const path = new URL(request.url).pathname
  let code = 200, data = []
  if (request.method !== 'GET') writes.push({ path, method: request.method })
  if (path === '/api/projects' && request.method === 'POST') {
    code = failCreate ? 503 : 201; failCreate = false; data = project
  } else if (path.endsWith('/imports/preview')) {
    code = failScan ? 503 : 200; failScan = false; data = preview
  } else if (path.endsWith('/confirm')) {
    code = failSave ? 503 : 200; failSave = false
    if (code === 200) saved = true
    data = { import_id: 'import-ui', message_count: 1240, created: true }
  } else if (request.method !== 'GET') { code = 403; data = { detail: '模拟环境禁止此写操作' } }
  else if (path.endsWith('/journey')) data = {
    project_id: project.id, stages: [], tasks: [], branches: [], processing: false,
    next_action: { label: '导入记录', action: 'import', object_id: null },
    imports: saved ? [{ id: 'import-ui', preview_id: preview.id, message_count: 1240 }] : [],
    participants: saved ? [{ id: 'self-ui', name: '小林', role: 'self' }, { id: 'target-ui', name: '小周', role: 'target' }] : [],
    time_range: preview.time_range, graph: saved ? { status: 'compiling_profile' } : null, publication: null,
  }
  else if (path.endsWith('/world-profile/status')) data = { status: 'compiling_profile', build_progress: { stage: 'compiling_profile', completed_questions: 2, question_count: 7, progress: 0.28 } }
  else if (path.endsWith('/world-profile')) { code = 404; data = { detail: '尚未发布' } }
  else if (path.endsWith('/annotations/summary')) data = { eligible: 0, annotated: 0, succeeded: 0, failed: 0, needs_review: 0, approved: 0, blocked: 0 }
  await send('Fetch.fulfillRequest', { requestId, responseCode: code, responseHeaders: [{ name: 'Content-Type', value: 'application/json' }], body: Buffer.from(JSON.stringify(data)).toString('base64') })
}
socket.onmessage = event => {
  const message = JSON.parse(event.data)
  if (message.id) { const call = pending.get(message.id); pending.delete(message.id); message.error ? call?.reject(Error(message.error.message)) : call?.resolve(message.result) }
  else if (message.method === 'Fetch.requestPaused') void respond(message.params).catch(e => errors.push(String(e)))
  else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text)
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
  if (result.exceptionDetails) throw Error(result.exceptionDetails.text)
  return result.result.value
}
async function until(expression) {
  const end = Date.now() + 15000
  while (Date.now() < end) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 100)) }
  throw Error(`页面未就绪：${expression}`)
}
async function click(label) {
  const expr = `Array.from(document.querySelectorAll('button,a')).find(e => e.textContent === ${JSON.stringify(label)})`
  await until(`Boolean(${expr})`); await evaluate(`${expr}.click()`)
}
async function field(id, value, select = false) {
  await evaluate(`(() => { const e=document.getElementById(${JSON.stringify(id)}); Object.getOwnPropertyDescriptor(${select ? 'HTMLSelectElement' : 'HTMLInputElement'}.prototype,'value').set.call(e,${JSON.stringify(value)}); e.dispatchEvent(new Event('${select ? 'change' : 'input'}',{bubbles:true})); })()`)
}
async function shot(width, name) {
  await new Promise(r => setTimeout(r, 200))
  const layout = await evaluate(`(() => {const e=document.querySelector('.setup-page'),m=document.querySelector('main');const a=e.getBoundingClientRect(),b=m.getBoundingClientRect(); return {overflow:document.documentElement.scrollWidth>innerWidth+2,width:a.width,offset:Math.abs((a.left+a.right)/2-(b.left+b.right)/2)};})()`)
  if (layout.overflow || layout.offset > 20) throw Error(JSON.stringify({ name, width, ...layout }))
  if (!columnWidths.has(width)) columnWidths.set(width, layout.width)
  if (Math.abs(columnWidths.get(width) - layout.width) > 1) throw Error(`同一视口的表单栏宽不一致：${name} ${layout.width}`)
  const image = await send('Page.captureScreenshot', { format: 'png' })
  await writeFile(join(directory, `${width}-${name}.png`), Buffer.from(image.data, 'base64'))
  console.log(JSON.stringify({ viewport: width, name, ...layout }))
}
try {
  await send('Page.enable'); await send('Runtime.enable')
  // 精确匹配 HTTP API，不能误拦 Vite 的 /src/api/client.ts 模块。
  await send('Fetch.enable', { patterns: [{ urlPattern: 'http://127.0.0.1:5175/api/*' }] })
  for (const width of [1440, 768, 390]) {
    saved = false; failCreate = true; failScan = true; failSave = true
    await send('Emulation.setDeviceMetricsOverride', { width, height: 1000, deviceScaleFactor: 1, mobile: false })
    await send('Page.navigate', { url: 'http://127.0.0.1:5175/projects/new' })
    await until(`Boolean(document.getElementById('project-name'))`)
    await evaluate('sessionStorage.clear()')
    await field('project-name', project.name); await shot(width, '创建')
    await click('继续导入聊天'); await until(`document.body.innerText.includes('项目创建失败')`); await shot(width, '创建失败')
    await click('继续导入聊天'); await until(`Boolean(document.getElementById('export-directory'))`)
    await shot(width, '选择目录')
    await evaluate(`(() => {const d=new DataTransfer(); d.items.add(new File(['mock only'],'chat.csv',{type:'text/csv'}));const e=document.getElementById('export-directory');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));})()`)
    await click('扫描并预览'); await until(`document.body.innerText.includes('扫描失败')`); await shot(width, '扫描失败')
    await click('扫描并预览'); await until(`document.body.innerText.includes('已解析，尚未保存')`); await shot(width, '解析预览')
    await click('记录已解析，确认人物与资料'); await until(`Boolean(document.getElementById('target-participant'))`)
    await field('self-participant', '小林', true); await field('target-participant', '小林', true); await shot(width, '相同人物校验')
    await field('target-participant', '小周', true); await shot(width, '确认人物')
    await click('确认人物并导入'); await until(`document.body.innerText.includes('导入失败')`); await shot(width, '保存失败')
    await click('确认人物并导入'); await until(`document.body.innerText.includes('已保存 1240 条记录')`); await shot(width, '保存完成')
    await click('查看人物背景'); await until(`document.querySelector('h1')?.textContent==='整理人物背景'`); await shot(width, '后台整理')
  }
  if (errors.length) throw Error(errors.join('\n'))
  console.log(JSON.stringify({ screenshots: directory, interceptedWrites: writes.length, realWrites: 0 }))
} finally { socket.close(); await fetch(`${host}/json/close/${target.id}`) }
