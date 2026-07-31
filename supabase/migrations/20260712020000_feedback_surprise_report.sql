begin;

-- SURPRISE-002 keeps the historical business_snapshot_v1 contract unchanged.
-- These additive RPCs expose only authoritative dimensions and post-reveal
-- surprise aggregates.  They never expose answer keys or ground-truth metrics.
create function public.feedback_report_surprise_questions(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_session_id text default null,
    p_attempt_id text default null
)
returns table (
    question_id text,
    question_version text,
    release_id text,
    family text,
    dataset_id text,
    question_type text,
    answered_attempt_count bigint,
    rating_count bigint,
    surprised_count bigint,
    not_surprised_count bigint,
    rating_coverage_rate numeric,
    observed_surprise_rate numeric,
    posterior_mean numeric,
    first_rating_at timestamptz,
    last_rating_at timestamptz
)
language sql
stable
security invoker
set search_path = ''
as $function$
with scoped_matched as (
    -- Do not apply the time window before deduplication: otherwise a later
    -- duplicate could become the first vote merely because the true first vote
    -- is outside the requested report window.
    select events.*
    from public.feedback_authoritative_events as events
    where events.registry_status = 'matched'
      and (
            p_release_id is null
            or events.authoritative_release_id = p_release_id
        )
      and (
            p_family is null
            or events.authoritative_family = p_family
        )
      and (
            p_question_type is null
            or events.authoritative_question_type = p_question_type
        )
      and (
            p_question_id is null
            or events.authoritative_question_id = p_question_id
        )
      and (
            p_session_id is null
            or events.session_id = p_session_id
        )
      and (
            p_attempt_id is null
            or events.report_attempt_id = p_attempt_id
        )
      and (p_from is null or p_to is null or p_from < p_to)
), matched_answers as (
    select events.*
    from scoped_matched as events
    where events.event_type = 'answer_submitted'
      and events.report_attempt_id is not null
), answered_attempts as (
    select distinct
        answers.session_id,
        answers.report_attempt_id,
        answers.authoritative_release_id,
        answers.authoritative_question_id,
        answers.authoritative_question_version,
        answers.authoritative_family,
        answers.authoritative_dataset_id,
        answers.authoritative_question_type
    from matched_answers as answers
    where (p_from is null or answers.occurred_at >= p_from)
      and (p_to is null or answers.occurred_at < p_to)
), eligible_reactions as (
    select reactions.*
    from scoped_matched as reactions
    where reactions.event_type = 'question_reaction_submitted'
      and reactions.report_attempt_id is not null
      and pg_catalog.jsonb_typeof(
            reactions.payload -> 'reaction'
        ) = 'string'
      and reactions.payload ->> 'reaction' = 'surprise'
      and pg_catalog.jsonb_typeof(reactions.payload -> 'value') = 'boolean'
      and pg_catalog.jsonb_typeof(reactions.payload -> 'timing') = 'string'
      and reactions.payload ->> 'timing' = 'after_reveal'
      and exists (
            select 1
            from matched_answers as answers
            where answers.session_id = reactions.session_id
              and answers.report_attempt_id = reactions.report_attempt_id
              and answers.authoritative_release_id
                    = reactions.authoritative_release_id
              and answers.authoritative_question_id
                    = reactions.authoritative_question_id
              and answers.authoritative_question_version
                    = reactions.authoritative_question_version
              and (
                    answers.occurred_at,
                    answers.sequence,
                    answers.event_id
                  ) < (
                    reactions.occurred_at,
                    reactions.sequence,
                    reactions.event_id
                  )
        )
), ranked_reactions as (
    select
        reactions.*,
        pg_catalog.row_number() over (
            partition by
                reactions.session_id,
                reactions.report_attempt_id,
                reactions.authoritative_release_id,
                reactions.authoritative_question_id,
                reactions.authoritative_question_version
            order by
                reactions.occurred_at,
                reactions.sequence,
                reactions.event_id
        ) as reaction_rank
    from eligible_reactions as reactions
), first_ratings as (
    select reactions.*
    from ranked_reactions as reactions
    join answered_attempts as attempts
        on attempts.session_id = reactions.session_id
       and attempts.report_attempt_id = reactions.report_attempt_id
       and attempts.authoritative_release_id
            = reactions.authoritative_release_id
       and attempts.authoritative_question_id
            = reactions.authoritative_question_id
       and attempts.authoritative_question_version
            = reactions.authoritative_question_version
    where reactions.reaction_rank = 1
      and (p_from is null or reactions.occurred_at >= p_from)
      and (p_to is null or reactions.occurred_at < p_to)
), answered_grouped as (
    select
        attempts.authoritative_question_id as question_id,
        attempts.authoritative_question_version as question_version,
        attempts.authoritative_release_id as release_id,
        attempts.authoritative_family as family,
        attempts.authoritative_dataset_id as dataset_id,
        attempts.authoritative_question_type as question_type,
        pg_catalog.count(*) as answered_attempt_count
    from answered_attempts as attempts
    group by
        attempts.authoritative_question_id,
        attempts.authoritative_question_version,
        attempts.authoritative_release_id,
        attempts.authoritative_family,
        attempts.authoritative_dataset_id,
        attempts.authoritative_question_type
), ratings_grouped as (
    select
        ratings.authoritative_question_id as question_id,
        ratings.authoritative_question_version as question_version,
        ratings.authoritative_release_id as release_id,
        pg_catalog.count(*) filter (
            where ratings.payload -> 'value' = 'true'::jsonb
        ) as surprised_count,
        pg_catalog.count(*) filter (
            where ratings.payload -> 'value' = 'false'::jsonb
        ) as not_surprised_count,
        pg_catalog.min(ratings.occurred_at) as first_rating_at,
        pg_catalog.max(ratings.occurred_at) as last_rating_at
    from first_ratings as ratings
    group by
        ratings.authoritative_question_id,
        ratings.authoritative_question_version,
        ratings.authoritative_release_id
), reported as (
    select
        answered.question_id,
        answered.question_version,
        answered.release_id,
        answered.family,
        answered.dataset_id,
        answered.question_type,
        answered.answered_attempt_count,
        (
            coalesce(ratings.surprised_count, 0)
            + coalesce(ratings.not_surprised_count, 0)
        )::bigint as rating_count,
        coalesce(ratings.surprised_count, 0)::bigint as surprised_count,
        coalesce(ratings.not_surprised_count, 0)::bigint
            as not_surprised_count,
        pg_catalog.round(
            (
                coalesce(ratings.surprised_count, 0)
                + coalesce(ratings.not_surprised_count, 0)
            )::numeric
            / nullif(answered.answered_attempt_count, 0),
            4
        ) as rating_coverage_rate,
        pg_catalog.round(
            coalesce(ratings.surprised_count, 0)::numeric
            / nullif(
                coalesce(ratings.surprised_count, 0)
                + coalesce(ratings.not_surprised_count, 0),
                0
            ),
            4
        ) as observed_surprise_rate,
        pg_catalog.round(
            (1 + coalesce(ratings.surprised_count, 0))::numeric
            / (
                2
                + coalesce(ratings.surprised_count, 0)
                + coalesce(ratings.not_surprised_count, 0)
            ),
            4
        ) as posterior_mean,
        ratings.first_rating_at,
        ratings.last_rating_at
    from answered_grouped as answered
    left join ratings_grouped as ratings
        on ratings.release_id = answered.release_id
       and ratings.question_id = answered.question_id
       and ratings.question_version = answered.question_version
)
select reported.*
from reported
order by
    reported.posterior_mean desc,
    reported.rating_count desc,
    reported.release_id,
    reported.question_id,
    reported.question_version;
$function$;

-- Quality counts are intentionally based on every raw reaction, including
-- unresolved registry claims.  Orphans have three mutually exclusive reasons:
-- registry mismatch, invalid payload shape, or no earlier authoritative answer.
-- A duplicate is only a later reaction which otherwise satisfies every rule.
create function public.feedback_report_surprise_quality(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_session_id text default null,
    p_attempt_id text default null
)
returns table (
    raw_reaction_count bigint,
    valid_reaction_count bigint,
    orphan_reaction_count bigint,
    duplicate_reaction_count bigint,
    registry_unmatched_reaction_count bigint,
    invalid_payload_reaction_count bigint,
    missing_prior_answer_reaction_count bigint,
    unknown_release_reaction_count bigint,
    counts_conserved boolean,
    orphan_breakdown_conserved boolean
)
language sql
stable
security invoker
set search_path = ''
as $function$
with raw_reactions as (
    select events.*
    from public.feedback_authoritative_events as events
    where events.event_type = 'question_reaction_submitted'
      and (
            p_release_id is null
            or coalesce(
                events.authoritative_release_id,
                events.claimed_release_id
            ) = p_release_id
        )
      and (
            p_family is null
            or coalesce(
                events.authoritative_family,
                events.claimed_family
            ) = p_family
        )
      and (
            p_question_type is null
            or coalesce(
                events.authoritative_question_type,
                events.claimed_question_type
            ) = p_question_type
        )
      and (
            p_question_id is null
            or coalesce(
                events.authoritative_question_id,
                events.question_id
            ) = p_question_id
        )
      and (
            p_session_id is null
            or events.session_id = p_session_id
        )
      and (
            p_attempt_id is null
            or events.report_attempt_id = p_attempt_id
        )
      and (p_from is null or p_to is null or p_from < p_to)
), reaction_checks as (
    select
        reactions.*,
        case
            when reactions.registry_status <> 'matched'
                then 'registry_unmatched'
            when reactions.report_attempt_id is null
              or pg_catalog.jsonb_typeof(
                    reactions.payload -> 'reaction'
                ) is distinct from 'string'
              or reactions.payload ->> 'reaction' is distinct from 'surprise'
              or pg_catalog.jsonb_typeof(
                    reactions.payload -> 'value'
                ) is distinct from 'boolean'
              or pg_catalog.jsonb_typeof(
                    reactions.payload -> 'timing'
                ) is distinct from 'string'
              or reactions.payload ->> 'timing'
                    is distinct from 'after_reveal'
                then 'invalid_payload'
            when not exists (
                select 1
                from public.feedback_authoritative_events as answers
                where answers.registry_status = 'matched'
                  and answers.event_type = 'answer_submitted'
                  and answers.report_attempt_id is not null
                  and answers.session_id = reactions.session_id
                  and answers.report_attempt_id = reactions.report_attempt_id
                  and answers.authoritative_release_id
                        = reactions.authoritative_release_id
                  and answers.authoritative_question_id
                        = reactions.authoritative_question_id
                  and answers.authoritative_question_version
                        = reactions.authoritative_question_version
                  and (
                        answers.occurred_at,
                        answers.sequence,
                        answers.event_id
                      ) < (
                        reactions.occurred_at,
                        reactions.sequence,
                        reactions.event_id
                      )
            ) then 'missing_prior_answer'
            else null
        end as orphan_reason
    from raw_reactions as reactions
), eligible_ranked as (
    select
        reactions.event_id,
        pg_catalog.row_number() over (
            partition by
                reactions.session_id,
                reactions.report_attempt_id,
                reactions.authoritative_release_id,
                reactions.authoritative_question_id,
                reactions.authoritative_question_version
            order by
                reactions.occurred_at,
                reactions.sequence,
                reactions.event_id
        ) as reaction_rank
    from reaction_checks as reactions
    where reactions.orphan_reason is null
), classified as (
    select
        reactions.*,
        case
            when reactions.orphan_reason is not null then 'orphan'
            when ranked.reaction_rank = 1 then 'valid'
            else 'duplicate'
        end as reaction_status
    from reaction_checks as reactions
    left join eligible_ranked as ranked
        on ranked.event_id = reactions.event_id
), windowed as (
    select reactions.*
    from classified as reactions
    where (p_from is null or reactions.occurred_at >= p_from)
      and (p_to is null or reactions.occurred_at < p_to)
), metrics as (
    select
        pg_catalog.count(*) as raw_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.reaction_status = 'valid'
        ) as valid_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.reaction_status = 'orphan'
        ) as orphan_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.reaction_status = 'duplicate'
        ) as duplicate_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.orphan_reason = 'registry_unmatched'
        ) as registry_unmatched_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.orphan_reason = 'invalid_payload'
        ) as invalid_payload_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.orphan_reason = 'missing_prior_answer'
        ) as missing_prior_answer_reaction_count,
        pg_catalog.count(*) filter (
            where reactions.registry_status = 'unknown_release'
        ) as unknown_release_reaction_count
    from windowed as reactions
)
select
    metrics.raw_reaction_count,
    metrics.valid_reaction_count,
    metrics.orphan_reaction_count,
    metrics.duplicate_reaction_count,
    metrics.registry_unmatched_reaction_count,
    metrics.invalid_payload_reaction_count,
    metrics.missing_prior_answer_reaction_count,
    metrics.unknown_release_reaction_count,
    metrics.raw_reaction_count
        = metrics.valid_reaction_count
        + metrics.orphan_reaction_count
        + metrics.duplicate_reaction_count as counts_conserved,
    metrics.orphan_reaction_count
        = metrics.registry_unmatched_reaction_count
        + metrics.invalid_payload_reaction_count
        + metrics.missing_prior_answer_reaction_count
        as orphan_breakdown_conserved
from metrics;
$function$;

revoke all on function public.feedback_report_surprise_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_surprise_quality(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_surprise_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_surprise_quality(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;

comment on function public.feedback_report_surprise_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Service-role-only SURPRISE-002 per-question ratings: first valid post-answer vote, Beta(1,1) posterior, no GT leakage.';
comment on function public.feedback_report_surprise_quality(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Service-role-only SURPRISE-002 reaction quality conservation, including unresolved registry claims, orphans, and duplicates.';

commit;
