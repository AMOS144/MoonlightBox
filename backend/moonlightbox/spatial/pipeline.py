from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.spatial.bundle_analysis import BundleAnalyzer, BundlePlaceMention
from moonlightbox.spatial.bundles import build_bundles
from moonlightbox.spatial.config import SpatialPipelineConfig
from moonlightbox.spatial.episodes import (
    EpisodeManifest,
    EpisodeMessage,
    build_episode_manifests,
    to_episode_message,
)
from moonlightbox.spatial.graph_builder import build_place_graph_snapshot
from moonlightbox.spatial.models import (
    PlaceGraphSnapshot,
    PlaceMention,
    SpatialAnalysisBundle,
    SpatialAnalysisRun,
    SpatialConversationEpisode,
    SpatialEpisodeCandidate,
    VisitEpisode,
    VisitObservation,
)
from moonlightbox.spatial.resolution import PlaceResolutionService, normalize_place_alias
from moonlightbox.spatial.retrieval import EpisodeHit, SpatialEpisodeIndex, load_query_bank
from moonlightbox.spatial.visits import build_visit_episodes, build_visit_observations


class SpatialPipeline:
    def __init__(
        self,
        *,
        config: SpatialPipelineConfig,
        index: SpatialEpisodeIndex,
        analyzer: BundleAnalyzer,
        resolver: PlaceResolutionService,
    ) -> None:
        self._config = config
        self._index = index
        self._analyzer = analyzer
        self._resolver = resolver

    def run(
        self,
        session: Session,
        *,
        project_id: str,
        import_id: str,
    ) -> SpatialAnalysisRun:
        target = session.scalar(
            select(Participant).where(
                Participant.project_id == project_id,
                Participant.role == "target",
            )
        )
        if target is None:
            raise ValueError("项目没有 target 参与者")
        run = self._get_or_create_run(
            session,
            project_id=project_id,
            import_id=import_id,
            target_person_id=target.id,
        )
        if run.status == "completed":
            return run
        session.commit()
        try:
            run.status = "running"
            run.stage = "episode_segmentation"
            messages = self._load_messages(session, project_id, import_id)
            manifests = build_episode_manifests(
                messages,
                project_id=project_id,
                import_id=import_id,
                segmentation_version=self._config.segmentation_version,
                episode_gap=self._config.episode_gap,
                character_budget=self._config.episode_character_budget,
                message_limit=self._config.episode_message_limit,
            )
            self._persist_episodes(session, run, manifests)

            run.stage = "episode_retrieval"
            self._index.replace(manifests)
            query_bank_version, queries = load_query_bank()
            if query_bank_version != self._config.query_bank_version:
                raise ValueError("空间 query bank 版本与流水线配置不一致")
            hits = self._index.retrieve(
                manifests,
                queries,
                top_k_per_partition=self._config.query_top_k_per_month,
                minimum_similarity=self._config.query_minimum_similarity,
            )
            self._persist_candidates(session, run.id, hits)

            run.stage = "bundle_analysis"
            bundles = build_bundles(
                manifests,
                hits,
                neighbor_gap=self._config.neighbor_gap,
                character_budget=self._config.episode_character_budget,
            )
            message_by_id = {message.id: message for message in messages}
            episode_by_id = {episode.id: episode for episode in manifests}
            for manifest in bundles:
                bundle = self._get_or_create_bundle(session, run.id, manifest)
                if bundle.status == "completed":
                    continue
                session.execute(delete(PlaceMention).where(PlaceMention.bundle_id == bundle.id))
                bundle.status = "analyzing"
                bundle_messages = [
                    message_by_id[message_id]
                    for episode_id in manifest.episode_ids
                    for message_id in episode_by_id[episode_id].message_ids
                ]
                result = self._analyzer.analyze(
                    messages=bundle_messages,
                    target_person_id=target.id,
                    run_id=run.id,
                    bundle_id=bundle.id,
                )
                bundle.analyzer_method = result.method
                bundle.analyzer_version = result.version
                for extracted in result.mentions:
                    self._persist_mention(
                        session,
                        run=run,
                        bundle=bundle,
                        extracted=extracted,
                        message_by_id=message_by_id,
                    )
                bundle.status = "completed"
                session.flush()

            run.stage = "entity_resolution"
            resolutions = self._resolver.resolve_run(
                session,
                run_id=run.id,
                project_id=project_id,
                target_person_id=target.id,
            )
            run.stage = "visit_aggregation"
            session.execute(
                delete(PlaceGraphSnapshot).where(PlaceGraphSnapshot.run_id == run.id)
            )
            session.execute(delete(VisitEpisode).where(VisitEpisode.run_id == run.id))
            session.execute(
                delete(VisitObservation).where(VisitObservation.run_id == run.id)
            )
            observations = build_visit_observations(
                session,
                run_id=run.id,
                target_person_id=target.id,
            )
            visit_episodes = build_visit_episodes(
                session,
                run_id=run.id,
                subject_id=target.id,
            )
            run.stage = "graph_projection"
            snapshot = build_place_graph_snapshot(
                session,
                project_id=project_id,
                run_id=run.id,
                person_id=target.id,
                graph_version=self._config.graph_version,
                cutoff_at=max((message.timestamp for message in messages), default=None),
            )
            run.statistics = {
                "message_count": len(messages),
                "episode_count": len(manifests),
                "candidate_episode_count": len(hits),
                "bundle_count": len(bundles),
                "mention_count": session.scalar(
                    select(func.count(PlaceMention.id)).where(PlaceMention.run_id == run.id)
                )
                or 0,
                "resolution_count": len(resolutions),
                "visit_observation_count": len(observations),
                "visit_episode_count": len(visit_episodes),
                "snapshot_id": snapshot.id,
            }
            run.status = "completed"
            run.stage = "completed"
            run.completed_at = datetime.now(UTC)
            session.commit()
            return run
        except Exception as error:
            session.rollback()
            failed = session.get(SpatialAnalysisRun, run.id)
            if failed is not None:
                failed.status = "failed"
                failed.error_code = type(error).__name__
                failed.error_message = str(error)[:2000]
                session.commit()
            raise

    def _get_or_create_run(
        self,
        session: Session,
        *,
        project_id: str,
        import_id: str,
        target_person_id: str,
    ) -> SpatialAnalysisRun:
        fingerprint = self._config.fingerprint()
        run = session.scalar(
            select(SpatialAnalysisRun).where(
                SpatialAnalysisRun.import_id == import_id,
                SpatialAnalysisRun.config_hash == fingerprint,
            )
        )
        if run is None:
            run = SpatialAnalysisRun(
                project_id=project_id,
                import_id=import_id,
                target_person_id=target_person_id,
                config=self._config.snapshot(),
                config_hash=fingerprint,
            )
            session.add(run)
            session.flush()
        return run

    @staticmethod
    def _load_messages(
        session: Session,
        project_id: str,
        import_id: str,
    ) -> list[EpisodeMessage]:
        rows = session.execute(
            select(Message, Participant)
            .join(Participant, Participant.id == Message.participant_id)
            .where(Message.project_id == project_id, Message.import_id == import_id)
            .order_by(Message.timestamp, Message.id)
        ).all()
        return [to_episode_message(message, participant) for message, participant in rows]

    @staticmethod
    def _persist_episodes(
        session: Session,
        run: SpatialAnalysisRun,
        episodes: list[EpisodeManifest],
    ) -> None:
        for index, episode in enumerate(episodes):
            previous_id = episodes[index - 1].id if index > 0 else None
            next_id = episodes[index + 1].id if index + 1 < len(episodes) else None
            existing = session.get(SpatialConversationEpisode, episode.id)
            if existing is None:
                session.add(
                    SpatialConversationEpisode(
                        id=episode.id,
                        project_id=episode.project_id,
                        import_id=episode.import_id,
                        created_run_id=run.id,
                        message_ids=list(episode.message_ids),
                        participant_ids=list(episode.participant_ids),
                        media_asset_ids=list(episode.media_asset_ids),
                        index_text=episode.index_text,
                        started_at=episode.started_at,
                        ended_at=episode.ended_at,
                        segmentation_version=episode.segmentation_version,
                        input_hash=episode.input_hash,
                        content_hash=episode.content_hash,
                        time_partition=episode.time_partition,
                        has_structured_location=episode.has_structured_location,
                        previous_episode_id=previous_id,
                        next_episode_id=next_id,
                    )
                )
            elif existing.content_hash != episode.content_hash:
                raise ValueError("相同 Episode ID 对应了不同内容")
        session.flush()

    @staticmethod
    def _persist_candidates(session: Session, run_id: str, hits: list[EpisodeHit]) -> None:
        for value in hits:
            existing = session.scalar(
                select(SpatialEpisodeCandidate).where(
                    SpatialEpisodeCandidate.run_id == run_id,
                    SpatialEpisodeCandidate.episode_id == value.episode_id,
                )
            )
            if existing is None:
                session.add(
                    SpatialEpisodeCandidate(
                        run_id=run_id,
                        episode_id=value.episode_id,
                        query_matches=value.query_matches,
                        max_similarity=value.max_similarity,
                        time_partition=value.time_partition,
                        forced_reason=value.forced_reason,
                    )
                )
            else:
                existing.query_matches = value.query_matches
                existing.max_similarity = value.max_similarity
                existing.forced_reason = value.forced_reason
        session.flush()

    @staticmethod
    def _get_or_create_bundle(
        session: Session,
        run_id: str,
        manifest: object,
    ) -> SpatialAnalysisBundle:
        from moonlightbox.spatial.bundles import BundleManifest

        if not isinstance(manifest, BundleManifest):
            raise TypeError("Bundle manifest 类型无效")
        bundle = session.scalar(
            select(SpatialAnalysisBundle).where(
                SpatialAnalysisBundle.run_id == run_id,
                SpatialAnalysisBundle.bundle_hash == manifest.bundle_hash,
            )
        )
        if bundle is None:
            bundle = SpatialAnalysisBundle(
                run_id=run_id,
                bundle_hash=manifest.bundle_hash,
                episode_ids=list(manifest.episode_ids),
                candidate_episode_ids=list(manifest.candidate_episode_ids),
                message_ids=list(manifest.message_ids),
                started_at=manifest.started_at,
                ended_at=manifest.ended_at,
            )
            session.add(bundle)
            session.flush()
        return bundle

    @staticmethod
    def _persist_mention(
        session: Session,
        *,
        run: SpatialAnalysisRun,
        bundle: SpatialAnalysisBundle,
        extracted: BundlePlaceMention,
        message_by_id: dict[str, EpisodeMessage],
    ) -> None:
        message = message_by_id[extracted.message_id]
        subject_id = extracted.subject if extracted.subject in {
            item.participant_id for item in message_by_id.values()
        } else None
        timestamp = extracted.time_start or message.timestamp
        mention = PlaceMention(
            project_id=run.project_id,
            import_id=run.import_id,
            run_id=run.id,
            bundle_id=bundle.id,
            message_id=message.id,
            span_start=extracted.span_start,
            span_end=extracted.span_end,
            raw_text=extracted.raw_text,
            normalized_text=normalize_place_alias(extracted.raw_text),
            mention_type=extracted.mention_type,
            type_distribution={extracted.mention_type: 1.0},
            candidate_sources=["local_bundle_analysis"],
            speaker_id=message.participant_id,
            subject_id=subject_id,
            subject_resolution="explicit" if subject_id else "unresolved",
            relation=extracted.relation,
            movement_phase=extracted.movement_phase,
            assertion_mode=extracted.assertion_mode,
            time_start_lower=timestamp,
            time_start_upper=timestamp,
            time_end_lower=extracted.time_end,
            time_end_upper=extracted.time_end,
            time_precision=extracted.time_precision,
            context_message_ids=bundle.message_ids,
            evidence_message_ids=extracted.evidence_message_ids,
            extractor_method=(bundle.analyzer_method or "bundle_analyzer"),
            raw_score=extracted.confidence,
            confidence=extracted.confidence,
        )
        session.add(mention)
