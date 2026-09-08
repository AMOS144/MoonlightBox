from __future__ import annotations

import argparse

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.quoted_sender_preprocess import preprocess_quoted_senders
from moonlightbox.world.jobs import load_world_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    args = parser.parse_args()
    database = Database(Settings().database_url)
    with next(database.session()) as session:
        observations = preprocess_quoted_senders(
            load_world_messages(session, project_id=args.project_id)
        )
    print(f"quoted_sender_observations={len(observations)}")
    for item in observations:
        target = f"{item.participant_name} ({item.participant_role})" if item.participant_name else "-"
        print(f"{item.alias}\t{item.status}\t{target}\tquotes={len(item.quoted_message_ids)}\tmessages={len(item.message_ids)}")


if __name__ == "__main__":
    main()
