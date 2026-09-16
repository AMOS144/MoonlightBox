"""项目任务归属：兼容旧 Runtime 仅保存 branch_id 的任务，不修改历史记录。"""

from sqlalchemy import and_, or_, select

from .models import Job


def project_job_condition(project_id):
    from moonlightbox.runtime_v1.branch_models import Branch

    explicit = Job.payload["project_id"].as_string()
    return or_(
        explicit == project_id,
        and_(
            explicit.is_(None),
            Job.payload["branch_id"].as_string().in_(
                select(Branch.id).where(Branch.project_id == project_id)
            ),
        ),
    )
