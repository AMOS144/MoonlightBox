// 只有后台确实会自行推进的阶段才需要轮询。`awaiting_*_review` 是用户停下来
// 审核内容的稳定状态；继续轮询会让整页频繁重渲染，长页面还会表现为滚动位置回到顶部。
export function shouldPollWorldGraphStatus(status: string | undefined): boolean {
  return ['building', 'profile_compilation_queued', 'compiling_profile'].includes(status ?? '')
}
