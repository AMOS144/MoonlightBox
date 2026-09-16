"""显式更新 latest_profile 分支到已发布画像；调用前需先停止旧版 Worker。"""

import argparse
import json

from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.runtime_v1.service import RuntimeService
from sqlalchemy import select
from sqlalchemy.orm import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--branch-id", required=True)
    args = parser.parse_args()
    database = Database(args.database_url)
    with Session(database.engine) as session:
        active = session.scalars(select(Job).where(Job.status == "running"))
        if any(job.payload.get("branch_id") == args.branch_id for job in active):
            raise RuntimeError("分支任务仍在执行，请等待或安全停止后更新")
        result = RuntimeService(session).refresh_published_background(
            args.project_id, args.branch_id
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
