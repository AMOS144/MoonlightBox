/** 项目数据契约与页面组件分离，其他功能不依赖项目列表页面。 */
export type Project = {
  id: string
  name: string
  status: string
  created_at: string
  updated_at: string
  target_name?: string | null
  target_avatar_asset_id?: string | null
}
