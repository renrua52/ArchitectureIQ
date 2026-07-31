begin;

-- REPORT-002 session/attempt drilldown is a forward-only cutover.  Drop the
-- atomic snapshot first because it depends on every business RPC.  Recreate
-- one exact signature per function: retaining defaulted overloads makes
-- PostgREST function selection ambiguous.
drop function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer
);
drop function public.feedback_report_summary(text, text, text, text, timestamptz, timestamptz);
drop function public.feedback_report_sessions(text, text, text, text, timestamptz, timestamptz);
drop function public.feedback_report_questions(text, text, text, text, timestamptz, timestamptz);
drop function public.feedback_report_comments(text, text, text, text, timestamptz, timestamptz, text);
drop function public.feedback_report_answers(text, text, text, text, timestamptz, timestamptz);
drop function public.feedback_report_proposals(text, text, text, text, timestamptz, timestamptz);

-- New parameters are appended after each historical positional signature.
-- Existing callers can omit them and receive the unfiltered behavior.
create function public.feedback_report_summary(
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
    event_count bigint,
    first_event_at timestamptz,
    last_event_at timestamptz,
    session_count bigint,
    attempt_count bigint,
    solve_attempt_count bigint,
    answered_attempt_count bigint,
    question_count bigint,
    answer_count bigint,
    known_answer_count bigint,
    correct_answer_count bigint,
    incorrect_answer_count bigint,
    unknown_answer_count bigint,
    accuracy numeric,
    proposal_count bigint,
    rejected_setting_count bigint,
    completed_run_count bigint,
    failed_run_count bigint,
    comment_count bigint,
    attempts_with_proposal bigint,
    proposal_usage_rate numeric,
    ingestion_failure_rate numeric,
    ingestion_failure_rate_available boolean
)
language sql
stable
security invoker
set search_path = ''
as $function$
with filtered as (
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
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
), attempts as (
    select distinct session_id, report_attempt_id
    from filtered
), solve_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type in (
        'answer_submitted',
        'custom_setting_proposed',
        'custom_setting_rejected',
        'custom_run_completed',
        'custom_run_failed'
    )
), answered_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type = 'answer_submitted'
), proposal_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type = 'custom_setting_proposed'
), metrics as (
    select
        pg_catalog.count(*) as event_count,
        pg_catalog.min(occurred_at) as first_event_at,
        pg_catalog.max(occurred_at) as last_event_at,
        pg_catalog.count(distinct session_id) as session_count,
        (select pg_catalog.count(*) from attempts) as attempt_count,
        (select pg_catalog.count(*) from solve_attempts) as solve_attempt_count,
        (select pg_catalog.count(*) from answered_attempts)
            as answered_attempt_count,
        pg_catalog.count(distinct (
            authoritative_question_id,
            authoritative_question_version
        )) as question_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
        ) as answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct is not null
        ) as known_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct
        ) as correct_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct = false
        ) as incorrect_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct is null
        ) as unknown_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'custom_setting_proposed'
        ) as proposal_count,
        pg_catalog.count(*) filter (
            where event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        pg_catalog.count(*) filter (
            where event_type = 'custom_run_completed'
        ) as completed_run_count,
        pg_catalog.count(*) filter (
            where event_type = 'custom_run_failed'
        ) as failed_run_count,
        pg_catalog.count(*) filter (
            where event_type = 'comment_submitted'
        ) as comment_count,
        (select pg_catalog.count(*) from proposal_attempts)
            as attempts_with_proposal
    from filtered
)
select
    metrics.event_count,
    metrics.first_event_at,
    metrics.last_event_at,
    metrics.session_count,
    metrics.attempt_count,
    metrics.solve_attempt_count,
    metrics.answered_attempt_count,
    metrics.question_count,
    metrics.answer_count,
    metrics.known_answer_count,
    metrics.correct_answer_count,
    metrics.incorrect_answer_count,
    metrics.unknown_answer_count,
    pg_catalog.round(
        metrics.correct_answer_count::numeric
        / nullif(metrics.known_answer_count, 0),
        4
    ) as accuracy,
    metrics.proposal_count,
    metrics.rejected_setting_count,
    metrics.completed_run_count,
    metrics.failed_run_count,
    metrics.comment_count,
    metrics.attempts_with_proposal,
    pg_catalog.round(
        metrics.attempts_with_proposal::numeric
        / nullif(metrics.solve_attempt_count, 0),
        4
    ) as proposal_usage_rate,
    null::numeric as ingestion_failure_rate,
    false as ingestion_failure_rate_available
from metrics;
$function$;

create function public.feedback_report_sessions(
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
    session_id text,
    attempt_id text,
    started_at timestamptz,
    last_event_at timestamptz,
    first_received_at timestamptz,
    last_received_at timestamptz,
    release_ids text[],
    families text[],
    question_types text[],
    event_count bigint,
    question_count bigint,
    answer_count bigint,
    known_answer_count bigint,
    correct_answer_count bigint,
    incorrect_answer_count bigint,
    unknown_answer_count bigint,
    accuracy numeric,
    proposal_count bigint,
    rejected_setting_count bigint,
    completed_run_count bigint,
    failed_run_count bigint,
    comment_count bigint
)
language sql
stable
security invoker
set search_path = ''
as $function$
with filtered as (
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
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
), grouped as (
    select
        filtered.session_id,
        filtered.report_attempt_id as attempt_id,
        pg_catalog.min(filtered.occurred_at) as started_at,
        pg_catalog.max(filtered.occurred_at) as last_event_at,
        pg_catalog.min(filtered.received_at) as first_received_at,
        pg_catalog.max(filtered.received_at) as last_received_at,
        coalesce(
            pg_catalog.array_agg(
                distinct filtered.authoritative_release_id
                order by filtered.authoritative_release_id
            ),
            '{}'::text[]
        ) as release_ids,
        coalesce(
            pg_catalog.array_agg(
                distinct filtered.authoritative_family
                order by filtered.authoritative_family
            ),
            '{}'::text[]
        ) as families,
        coalesce(
            pg_catalog.array_agg(
                distinct filtered.authoritative_question_type
                order by filtered.authoritative_question_type
            ),
            '{}'::text[]
        ) as question_types,
        pg_catalog.count(*) as event_count,
        pg_catalog.count(distinct (
            filtered.authoritative_question_id,
            filtered.authoritative_question_version
        )) as question_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct is not null
        ) as known_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct
        ) as correct_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct = false
        ) as incorrect_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct is null
        ) as unknown_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as proposal_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_run_completed'
        ) as completed_run_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_run_failed'
        ) as failed_run_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'comment_submitted'
        ) as comment_count
    from filtered
    group by filtered.session_id, filtered.report_attempt_id
)
select
    grouped.session_id,
    grouped.attempt_id,
    grouped.started_at,
    grouped.last_event_at,
    grouped.first_received_at,
    grouped.last_received_at,
    grouped.release_ids,
    grouped.families,
    grouped.question_types,
    grouped.event_count,
    grouped.question_count,
    grouped.answer_count,
    grouped.known_answer_count,
    grouped.correct_answer_count,
    grouped.incorrect_answer_count,
    grouped.unknown_answer_count,
    pg_catalog.round(
        grouped.correct_answer_count::numeric
        / nullif(grouped.known_answer_count, 0),
        4
    ) as accuracy,
    grouped.proposal_count,
    grouped.rejected_setting_count,
    grouped.completed_run_count,
    grouped.failed_run_count,
    grouped.comment_count
from grouped
order by grouped.last_event_at desc, grouped.session_id, grouped.attempt_id;
$function$;

create function public.feedback_report_questions(
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
    first_event_at timestamptz,
    last_event_at timestamptz,
    event_count bigint,
    session_count bigint,
    attempt_count bigint,
    solve_attempt_count bigint,
    answered_attempt_count bigint,
    answer_count bigint,
    known_answer_count bigint,
    correct_answer_count bigint,
    incorrect_answer_count bigint,
    unknown_answer_count bigint,
    accuracy numeric,
    proposal_count bigint,
    rejected_setting_count bigint,
    completed_run_count bigint,
    failed_run_count bigint,
    comment_count bigint,
    attempts_with_proposal bigint,
    proposal_usage_rate numeric
)
language sql
stable
security invoker
set search_path = ''
as $function$
with filtered as (
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
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
), grouped as (
    select
        filtered.authoritative_question_id as question_id,
        filtered.authoritative_question_version as question_version,
        filtered.authoritative_release_id as release_id,
        filtered.authoritative_family as family,
        filtered.authoritative_dataset_id as dataset_id,
        filtered.authoritative_question_type as question_type,
        pg_catalog.min(filtered.occurred_at) as first_event_at,
        pg_catalog.max(filtered.occurred_at) as last_event_at,
        pg_catalog.count(*) as event_count,
        pg_catalog.count(distinct filtered.session_id) as session_count,
        pg_catalog.count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) as attempt_count,
        pg_catalog.count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) filter (
            where filtered.event_type in (
                'answer_submitted',
                'custom_setting_proposed',
                'custom_setting_rejected',
                'custom_run_completed',
                'custom_run_failed'
            )
        ) as solve_attempt_count,
        pg_catalog.count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answered_attempt_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct is not null
        ) as known_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct
        ) as correct_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct = false
        ) as incorrect_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.authoritative_is_correct is null
        ) as unknown_answer_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as proposal_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_run_completed'
        ) as completed_run_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'custom_run_failed'
        ) as failed_run_count,
        pg_catalog.count(*) filter (
            where filtered.event_type = 'comment_submitted'
        ) as comment_count,
        pg_catalog.count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as attempts_with_proposal
    from filtered
    group by
        filtered.authoritative_question_id,
        filtered.authoritative_question_version,
        filtered.authoritative_release_id,
        filtered.authoritative_family,
        filtered.authoritative_dataset_id,
        filtered.authoritative_question_type
)
select
    grouped.question_id,
    grouped.question_version,
    grouped.release_id,
    grouped.family,
    grouped.dataset_id,
    grouped.question_type,
    grouped.first_event_at,
    grouped.last_event_at,
    grouped.event_count,
    grouped.session_count,
    grouped.attempt_count,
    grouped.solve_attempt_count,
    grouped.answered_attempt_count,
    grouped.answer_count,
    grouped.known_answer_count,
    grouped.correct_answer_count,
    grouped.incorrect_answer_count,
    grouped.unknown_answer_count,
    pg_catalog.round(
        grouped.correct_answer_count::numeric
        / nullif(grouped.known_answer_count, 0),
        4
    ) as accuracy,
    grouped.proposal_count,
    grouped.rejected_setting_count,
    grouped.completed_run_count,
    grouped.failed_run_count,
    grouped.comment_count,
    grouped.attempts_with_proposal,
    pg_catalog.round(
        grouped.attempts_with_proposal::numeric
        / nullif(grouped.solve_attempt_count, 0),
        4
    ) as proposal_usage_rate
from grouped
order by grouped.answer_count desc, grouped.question_id, grouped.question_version;
$function$;

create function public.feedback_report_comments(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_category text default null,
    p_session_id text default null,
    p_attempt_id text default null
)
returns table (
    event_id text,
    occurred_at timestamptz,
    received_at timestamptz,
    session_id text,
    attempt_id text,
    question_id text,
    question_version text,
    release_id text,
    family text,
    question_type text,
    category text,
    comment_text text
)
language sql
stable
security invoker
set search_path = ''
as $function$
select
    events.event_id,
    events.occurred_at,
    events.received_at,
    events.session_id,
    events.report_attempt_id,
    events.authoritative_question_id,
    events.authoritative_question_version,
    events.authoritative_release_id,
    events.authoritative_family,
    events.authoritative_question_type,
    events.payload ->> 'category',
    events.payload ->> 'text'
from public.feedback_authoritative_events as events
where events.registry_status = 'matched'
  and events.event_type = 'comment_submitted'
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
  and (p_from is null or events.occurred_at >= p_from)
  and (p_to is null or events.occurred_at < p_to)
  and (p_category is null or events.payload ->> 'category' = p_category)
  and (p_from is null or p_to is null or p_from < p_to)
order by events.occurred_at desc, events.event_id;
$function$;

create function public.feedback_report_answers(
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
    event_id text,
    occurred_at timestamptz,
    received_at timestamptz,
    session_id text,
    attempt_id text,
    question_id text,
    question_version text,
    release_id text,
    family text,
    dataset_id text,
    question_type text,
    selected_letter text,
    client_selected_candidate_id text,
    selected_candidate_id text,
    answer_status text,
    is_correct boolean,
    client_is_correct boolean,
    client_context_mismatch boolean,
    client_correctness_mismatch boolean
)
language sql
stable
security invoker
set search_path = ''
as $function$
select
    events.event_id,
    events.occurred_at,
    events.received_at,
    events.session_id,
    events.report_attempt_id as attempt_id,
    events.authoritative_question_id as question_id,
    events.authoritative_question_version as question_version,
    events.authoritative_release_id as release_id,
    events.authoritative_family as family,
    events.authoritative_dataset_id as dataset_id,
    events.authoritative_question_type as question_type,
    nullif(events.payload ->> 'selected_letter', '') as selected_letter,
    nullif(
        events.payload ->> 'selected_candidate_id',
        ''
    ) as client_selected_candidate_id,
    events.authoritative_selected_candidate_id as selected_candidate_id,
    events.answer_status,
    events.authoritative_is_correct as is_correct,
    case
        when pg_catalog.jsonb_typeof(events.payload -> 'is_correct') = 'boolean'
            then events.payload -> 'is_correct' = 'true'::jsonb
        else null
    end as client_is_correct,
    events.client_context_mismatch,
    events.client_correctness_mismatch
from public.feedback_authoritative_events as events
where events.registry_status = 'matched'
  and events.event_type = 'answer_submitted'
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
  and (p_from is null or events.occurred_at >= p_from)
  and (p_to is null or events.occurred_at < p_to)
  and (p_from is null or p_to is null or p_from < p_to)
order by events.occurred_at desc, events.event_id;
$function$;

create function public.feedback_report_proposals(
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
    event_id text,
    occurred_at timestamptz,
    received_at timestamptz,
    session_id text,
    attempt_id text,
    question_id text,
    question_version text,
    release_id text,
    family text,
    dataset_id text,
    question_type text,
    setting_status text,
    label text,
    setting_json text,
    inherited_from_json text,
    n_seeds integer,
    base_seed integer,
    error_type text
)
language sql
stable
security invoker
set search_path = ''
as $function$
select
    events.event_id,
    events.occurred_at,
    events.received_at,
    events.session_id,
    events.report_attempt_id as attempt_id,
    events.authoritative_question_id as question_id,
    events.authoritative_question_version as question_version,
    events.authoritative_release_id as release_id,
    events.authoritative_family as family,
    events.authoritative_dataset_id as dataset_id,
    events.authoritative_question_type as question_type,
    case events.event_type
        when 'custom_setting_proposed' then 'proposed'
        when 'custom_setting_rejected' then 'rejected'
    end as setting_status,
    events.payload ->> 'label' as label,
    case
        when pg_catalog.jsonb_typeof(events.payload -> 'setting') = 'object'
            then (events.payload -> 'setting')::text
        else null
    end as setting_json,
    case
        when pg_catalog.jsonb_typeof(
            events.payload -> 'inherited_from'
        ) = 'object'
            then (events.payload -> 'inherited_from')::text
        else null
    end as inherited_from_json,
    case
        when pg_catalog.jsonb_typeof(events.payload -> 'n_seeds') = 'number'
            then case
                when (events.payload ->> 'n_seeds')::numeric
                        = pg_catalog.trunc(
                            (events.payload ->> 'n_seeds')::numeric
                        )
                  and (events.payload ->> 'n_seeds')::numeric
                        between -2147483648 and 2147483647
                    then (
                        (events.payload ->> 'n_seeds')::numeric
                    )::integer
                else null
            end
        else null
    end as n_seeds,
    case
        when pg_catalog.jsonb_typeof(events.payload -> 'base_seed') = 'number'
            then case
                when (events.payload ->> 'base_seed')::numeric
                        = pg_catalog.trunc(
                            (events.payload ->> 'base_seed')::numeric
                        )
                  and (events.payload ->> 'base_seed')::numeric
                        between -2147483648 and 2147483647
                    then (
                        (events.payload ->> 'base_seed')::numeric
                    )::integer
                else null
            end
        else null
    end as base_seed,
    events.payload ->> 'error_type' as error_type
from public.feedback_authoritative_events as events
where events.registry_status = 'matched'
  and events.event_type in (
        'custom_setting_proposed',
        'custom_setting_rejected'
    )
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
  and (p_from is null or events.occurred_at >= p_from)
  and (p_to is null or events.occurred_at < p_to)
  and (p_from is null or p_to is null or p_from < p_to)
order by events.occurred_at desc, events.event_id;
$function$;

create function public.feedback_report_business_snapshot(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_limit integer default 200,
    p_session_id text default null,
    p_attempt_id text default null
)
returns table (
    snapshot_revision text,
    snapshot_at timestamptz,
    authority_revision text,
    business_reports_authoritative boolean,
    registered_release_count bigint,
    registered_question_count bigint,
    registered_choice_count bigint,
    detail_revision text,
    detail_reports_authoritative boolean,
    pages_json text
)
language sql
stable
security invoker
set search_path = ''
as $function$
with parameters as materialized (
    -- The protected Edge and Python clients enforce the same range.  Keeping
    -- the guard here makes an invalid direct service-role call return no
    -- singleton row, which the protected Edge fails closed.
    select
        p_limit as page_limit,
        p_session_id as session_id,
        p_attempt_id as attempt_id,
        p_release_id as release_id,
        p_family as family,
        p_question_type as question_type,
        p_question_id as question_id,
        p_from as from_at,
        p_to as to_at
    where p_limit between 1 and 1000
), authority as materialized (
    select status.*
    from parameters
    cross join lateral (
        select status.*
        from public.feedback_report_authority_status() as status
        limit parameters.page_limit
    ) as status
), byte_budgets as materialized (
    -- Each page admits only a deterministic ordered prefix of complete rows.
    -- A selected row is never shortened or projected differently.  Proposal
    -- rows receive 2.5 MiB because one accepted ingest request is at most
    -- 1 MiB and rendering object-valued setting fields as JSON text can escape
    -- their contents a second time.  The six page budgets sum to 3,604,480
    -- bytes, leaving 589,824 bytes of the 4 MiB pages document budget for JSON
    -- arrays, keys, totals, and other fixed envelope syntax.
    select
        65536::bigint as summary_page_bytes,
        262144::bigint as sessions_page_bytes,
        262144::bigint as questions_page_bytes,
        262144::bigint as answers_page_bytes,
        2621440::bigint as proposals_page_bytes,
        131072::bigint as comments_page_bytes,
        4194304::bigint as snapshot_pages_bytes
), page_definitions as materialized (
    select definitions.view_name, definitions.page_byte_budget
    from byte_budgets
    cross join lateral (
        values
            ('feedback_report_summary', summary_page_bytes),
            ('feedback_report_sessions', sessions_page_bytes),
            ('feedback_report_questions', questions_page_bytes),
            ('feedback_report_answers', answers_page_bytes),
            ('feedback_report_proposals', proposals_page_bytes),
            ('feedback_report_comments', comments_page_bytes)
    ) as definitions(view_name, page_byte_budget)
), summary_page_rows as materialized (
    select ranked.*
    from (
        select
            summary.*,
            pg_catalog.row_number() over () as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_summary(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            parameters.session_id,
            parameters.attempt_id
        ) as summary
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), session_page_rows as materialized (
    select ranked.*
    from (
        select
            sessions.*,
            pg_catalog.row_number() over (
                order by
                    sessions.last_event_at desc,
                    sessions.session_id asc,
                    sessions.attempt_id asc nulls first
            ) as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_sessions(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            parameters.session_id,
            parameters.attempt_id
        ) as sessions
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), question_page_rows as materialized (
    select ranked.*
    from (
        select
            questions.*,
            pg_catalog.row_number() over (
                order by
                    questions.answer_count desc,
                    questions.question_id asc,
                    questions.question_version asc,
                    questions.release_id asc nulls first,
                    questions.family asc nulls first,
                    questions.dataset_id asc nulls first,
                    questions.question_type asc nulls first
            ) as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_questions(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            parameters.session_id,
            parameters.attempt_id
        ) as questions
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), answer_page_rows as materialized (
    select ranked.*
    from (
        select
            answers.*,
            pg_catalog.row_number() over (
                order by answers.occurred_at desc, answers.event_id asc
            ) as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_answers(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            parameters.session_id,
            parameters.attempt_id
        ) as answers
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), proposal_page_rows as materialized (
    select ranked.*
    from (
        select
            proposals.*,
            pg_catalog.row_number() over (
                order by proposals.occurred_at desc, proposals.event_id asc
            ) as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_proposals(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            parameters.session_id,
            parameters.attempt_id
        ) as proposals
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), comment_page_rows as materialized (
    select ranked.*
    from (
        select
            comments.*,
            pg_catalog.row_number() over (
                order by comments.occurred_at desc, comments.event_id asc
            ) as snapshot_page_rank,
            pg_catalog.count(*) over () as snapshot_exact_total
        from parameters
        cross join lateral public.feedback_report_comments(
            parameters.release_id,
            parameters.family,
            parameters.question_type,
            parameters.question_id,
            parameters.from_at,
            parameters.to_at,
            null,
            parameters.session_id,
            parameters.attempt_id
        ) as comments
    ) as ranked
    where ranked.snapshot_page_rank <= (
        select parameters.page_limit from parameters
    )
), report_rows as not materialized (
    -- The six staging CTEs above require an O(N) scan/window for exact totals,
    -- but materialize at most p_limit complete rows each.  JSON conversion and
    -- byte measurement happen only after that bounded staging barrier.  The two
    -- reserved snapshot_* names are removed to reproduce each RPC row exactly.
    select
        'feedback_report_summary'::text as view_name,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total' as row_json,
        rows.snapshot_page_rank as page_rank,
        rows.snapshot_exact_total as exact_total
    from summary_page_rows as rows

    union all

    select
        'feedback_report_sessions'::text,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total',
        rows.snapshot_page_rank,
        rows.snapshot_exact_total
    from session_page_rows as rows

    union all

    select
        'feedback_report_questions'::text,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total',
        rows.snapshot_page_rank,
        rows.snapshot_exact_total
    from question_page_rows as rows

    union all

    select
        'feedback_report_answers'::text,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total',
        rows.snapshot_page_rank,
        rows.snapshot_exact_total
    from answer_page_rows as rows

    union all

    select
        'feedback_report_proposals'::text,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total',
        rows.snapshot_page_rank,
        rows.snapshot_exact_total
    from proposal_page_rows as rows

    union all

    select
        'feedback_report_comments'::text,
        pg_catalog.to_jsonb(rows)
            - 'snapshot_page_rank'
            - 'snapshot_exact_total',
        rows.snapshot_page_rank,
        rows.snapshot_exact_total
    from comment_page_rows as rows
), sized_rows as not materialized (
    -- Rank/total windows run over every matching row, but only the requested
    -- ordered prefix reaches JSON conversion and byte measurement above.  One
    -- provisional comma byte per row plus one fixed byte at the budget check
    -- exactly accounts for the n - 1 commas and two brackets of a non-empty
    -- JSON array.  No selected row is shortened, and no later small row can jump
    -- ahead of an oversized first row.
    select
        rows.view_name,
        rows.row_json,
        rows.page_rank,
        rows.exact_total,
        (
            pg_catalog.octet_length(
                pg_catalog.convert_to(rows.row_json::text, 'UTF8')
            ) + 1
        )::bigint as serialized_row_bytes
    from report_rows as rows
), budgeted_rows as not materialized (
    select
        rows.*,
        pg_catalog.sum(rows.serialized_row_bytes) over (
            partition by rows.view_name
            order by rows.page_rank
            rows between unbounded preceding and current row
        ) as cumulative_page_bytes
    from sized_rows as rows
), page_results as materialized (
    select
        definitions.view_name,
        coalesce(
            pg_catalog.jsonb_agg(
                rows.row_json
                order by rows.page_rank
            ) filter (
                where rows.cumulative_page_bytes + 1
                    <= definitions.page_byte_budget
            ),
            '[]'::jsonb
        ) as rows_json,
        coalesce(
            pg_catalog.max(rows.exact_total),
            0
        )::bigint as exact_total
    from page_definitions as definitions
    left join budgeted_rows as rows
        on rows.view_name = definitions.view_name
    group by definitions.view_name, definitions.page_byte_budget
), pages_document as materialized (
    select pg_catalog.jsonb_build_object(
        'feedback_report_summary',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_summary',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_summary'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_summary'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        ),
        'feedback_report_sessions',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_sessions',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_sessions'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_sessions'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        ),
        'feedback_report_questions',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_questions',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_questions'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_questions'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        ),
        'feedback_report_answers',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_answers',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_answers'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_answers'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        ),
        'feedback_report_proposals',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_proposals',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_proposals'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_proposals'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        ),
        'feedback_report_comments',
        pg_catalog.jsonb_build_object(
            'view', 'feedback_report_comments',
            'rows', (
                select results.rows_json
                from page_results as results
                where results.view_name = 'feedback_report_comments'
            ),
            'total', (
                select results.exact_total
                from page_results as results
                where results.view_name = 'feedback_report_comments'
            ),
            'limit', parameters.page_limit,
            'offset', 0
        )
    ) as document
    from parameters
)
select
    'business_snapshot_v1'::text as snapshot_revision,
    pg_catalog.statement_timestamp() as snapshot_at,
    authority.authority_revision,
    authority.business_reports_authoritative,
    authority.registered_release_count,
    authority.registered_question_count,
    authority.registered_choice_count,
    authority.detail_revision,
    authority.detail_reports_authoritative,
    pages_document.document::text as pages_json
from parameters
cross join authority
cross join byte_budgets
cross join pages_document
-- pages_json is embedded once more as a JSON string by PostgREST.  Valid JSON
-- text contains no raw control bytes, so that embedding adds at most one byte
-- for each existing quote or backslash: at most 2 * 4 MiB + two quote bytes.
-- The remaining response fields are fixed and small, leaving more than 2 MiB
-- below the strict Python client's 10 MiB response ceiling.
where pg_catalog.octet_length(
    pg_catalog.convert_to(pages_document.document::text, 'UTF8')
) <= byte_budgets.snapshot_pages_bytes;
$function$;

revoke all on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz, text, text
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer, text, text
) from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text, text, text
) to service_role;
grant execute on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz, text, text
) to service_role;
grant execute on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer, text, text
) to service_role;

comment on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Protected authoritative summary with optional exact session and attempt filters.';
comment on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Protected authoritative session rows with optional exact session and attempt filters.';
comment on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Protected authoritative question rows with optional exact session and attempt filters.';
comment on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text, text, text
) is 'Protected categorized comments with optional exact session and attempt filters.';
comment on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Protected answer details with optional exact session and attempt filters.';
comment on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz, text, text
) is 'Protected proposal details with optional exact session and attempt filters.';
comment on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer, text, text
) is 'Protected business_snapshot_v1 with optional exact session and attempt filters.';

commit;

