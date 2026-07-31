begin;

-- Protected REPORT-001 RPCs. Every filter is applied to feedback_events before
-- aggregation so Summary, Sessions, and Questions use the same selected fact
-- set. These functions are callable only by the service role held by the
-- feedback-report Edge Function.

create function public.feedback_report_summary(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null
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
    select
        events.*,
        nullif(events.payload ->> 'attempt_id', '') as report_attempt_id
    from public.feedback_events as events
    where (p_release_id is null
            or nullif(events.payload ->> 'release_id', '') = p_release_id)
      and (p_family is null
            or nullif(events.payload ->> 'family', '') = p_family)
      and (p_question_type is null
            or nullif(events.payload ->> 'question_type', '') = p_question_type)
      and (p_question_id is null or events.question_id = p_question_id)
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
),
attempts as (
    select distinct session_id, report_attempt_id
    from filtered
),
solve_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type in (
        'answer_submitted',
        'custom_setting_proposed',
        'custom_setting_rejected',
        'custom_run_completed',
        'custom_run_failed'
    )
),
answered_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type = 'answer_submitted'
),
proposal_attempts as (
    select distinct session_id, report_attempt_id
    from filtered
    where event_type = 'custom_setting_proposed'
),
metrics as (
    select
        count(*) as event_count,
        min(occurred_at) as first_event_at,
        max(occurred_at) as last_event_at,
        count(distinct session_id) as session_count,
        (select count(*) from attempts) as attempt_count,
        (select count(*) from solve_attempts) as solve_attempt_count,
        (select count(*) from answered_attempts) as answered_attempt_count,
        count(distinct (question_id, question_version)) as question_count,
        count(*) filter (
            where event_type = 'answer_submitted'
        ) as answer_count,
        count(*) filter (
            where event_type = 'answer_submitted'
              and jsonb_typeof(payload -> 'is_correct') = 'boolean'
        ) as known_answer_count,
        count(*) filter (
            where event_type = 'answer_submitted'
              and payload -> 'is_correct' = 'true'::jsonb
        ) as correct_answer_count,
        count(*) filter (
            where event_type = 'answer_submitted'
              and payload -> 'is_correct' = 'false'::jsonb
        ) as incorrect_answer_count,
        count(*) filter (
            where event_type = 'answer_submitted'
              and jsonb_typeof(payload -> 'is_correct') is distinct from 'boolean'
        ) as unknown_answer_count,
        count(*) filter (
            where event_type = 'custom_setting_proposed'
        ) as proposal_count,
        count(*) filter (
            where event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        count(*) filter (
            where event_type = 'custom_run_completed'
        ) as completed_run_count,
        count(*) filter (
            where event_type = 'custom_run_failed'
        ) as failed_run_count,
        count(*) filter (
            where event_type = 'comment_submitted'
        ) as comment_count,
        (select count(*) from proposal_attempts) as attempts_with_proposal
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
    round(
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
    round(
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
    p_to timestamptz default null
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
    select
        events.*,
        nullif(events.payload ->> 'attempt_id', '') as report_attempt_id,
        nullif(events.payload ->> 'release_id', '') as report_release_id,
        nullif(events.payload ->> 'family', '') as report_family,
        nullif(events.payload ->> 'question_type', '') as report_question_type
    from public.feedback_events as events
    where (p_release_id is null
            or nullif(events.payload ->> 'release_id', '') = p_release_id)
      and (p_family is null
            or nullif(events.payload ->> 'family', '') = p_family)
      and (p_question_type is null
            or nullif(events.payload ->> 'question_type', '') = p_question_type)
      and (p_question_id is null or events.question_id = p_question_id)
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
),
grouped as (
    select
        filtered.session_id,
        filtered.report_attempt_id as attempt_id,
        min(filtered.occurred_at) as started_at,
        max(filtered.occurred_at) as last_event_at,
        min(filtered.received_at) as first_received_at,
        max(filtered.received_at) as last_received_at,
        coalesce(
            array_agg(
                distinct filtered.report_release_id
                order by filtered.report_release_id
            ) filter (where filtered.report_release_id is not null),
            '{}'::text[]
        ) as release_ids,
        coalesce(
            array_agg(
                distinct filtered.report_family
                order by filtered.report_family
            ) filter (where filtered.report_family is not null),
            '{}'::text[]
        ) as families,
        coalesce(
            array_agg(
                distinct filtered.report_question_type
                order by filtered.report_question_type
            ) filter (where filtered.report_question_type is not null),
            '{}'::text[]
        ) as question_types,
        count(*) as event_count,
        count(distinct (filtered.question_id, filtered.question_version))
            as question_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and jsonb_typeof(filtered.payload -> 'is_correct') = 'boolean'
        ) as known_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.payload -> 'is_correct' = 'true'::jsonb
        ) as correct_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.payload -> 'is_correct' = 'false'::jsonb
        ) as incorrect_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and jsonb_typeof(filtered.payload -> 'is_correct')
                  is distinct from 'boolean'
        ) as unknown_answer_count,
        count(*) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as proposal_count,
        count(*) filter (
            where filtered.event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        count(*) filter (
            where filtered.event_type = 'custom_run_completed'
        ) as completed_run_count,
        count(*) filter (
            where filtered.event_type = 'custom_run_failed'
        ) as failed_run_count,
        count(*) filter (
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
    round(
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
    p_to timestamptz default null
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
    select
        events.*,
        nullif(events.payload ->> 'attempt_id', '') as report_attempt_id,
        nullif(events.payload ->> 'release_id', '') as report_release_id,
        nullif(events.payload ->> 'family', '') as report_family,
        nullif(events.payload ->> 'dataset_id', '') as report_dataset_id,
        nullif(events.payload ->> 'question_type', '') as report_question_type
    from public.feedback_events as events
    where (p_release_id is null
            or nullif(events.payload ->> 'release_id', '') = p_release_id)
      and (p_family is null
            or nullif(events.payload ->> 'family', '') = p_family)
      and (p_question_type is null
            or nullif(events.payload ->> 'question_type', '') = p_question_type)
      and (p_question_id is null or events.question_id = p_question_id)
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
),
grouped as (
    select
        filtered.question_id,
        filtered.question_version,
        filtered.report_release_id as release_id,
        filtered.report_family as family,
        filtered.report_dataset_id as dataset_id,
        filtered.report_question_type as question_type,
        min(filtered.occurred_at) as first_event_at,
        max(filtered.occurred_at) as last_event_at,
        count(*) as event_count,
        count(distinct filtered.session_id) as session_count,
        count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) as attempt_count,
        count(distinct (
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
        count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answered_attempt_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
        ) as answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and jsonb_typeof(filtered.payload -> 'is_correct') = 'boolean'
        ) as known_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.payload -> 'is_correct' = 'true'::jsonb
        ) as correct_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and filtered.payload -> 'is_correct' = 'false'::jsonb
        ) as incorrect_answer_count,
        count(*) filter (
            where filtered.event_type = 'answer_submitted'
              and jsonb_typeof(filtered.payload -> 'is_correct')
                  is distinct from 'boolean'
        ) as unknown_answer_count,
        count(*) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as proposal_count,
        count(*) filter (
            where filtered.event_type = 'custom_setting_rejected'
        ) as rejected_setting_count,
        count(*) filter (
            where filtered.event_type = 'custom_run_completed'
        ) as completed_run_count,
        count(*) filter (
            where filtered.event_type = 'custom_run_failed'
        ) as failed_run_count,
        count(*) filter (
            where filtered.event_type = 'comment_submitted'
        ) as comment_count,
        count(distinct (
            filtered.session_id,
            coalesce(filtered.report_attempt_id, '')
        )) filter (
            where filtered.event_type = 'custom_setting_proposed'
        ) as attempts_with_proposal
    from filtered
    group by
        filtered.question_id,
        filtered.question_version,
        filtered.report_release_id,
        filtered.report_family,
        filtered.report_dataset_id,
        filtered.report_question_type
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
    round(
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
    round(
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
    p_category text default null
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
with filtered as (
    select
        events.event_id,
        events.occurred_at,
        events.received_at,
        events.session_id,
        nullif(events.payload ->> 'attempt_id', '') as attempt_id,
        events.question_id,
        events.question_version,
        nullif(events.payload ->> 'release_id', '') as release_id,
        nullif(events.payload ->> 'family', '') as family,
        nullif(events.payload ->> 'question_type', '') as question_type,
        events.payload ->> 'category' as category,
        events.payload ->> 'text' as comment_text
    from public.feedback_events as events
    where events.event_type = 'comment_submitted'
      and (p_release_id is null
            or nullif(events.payload ->> 'release_id', '') = p_release_id)
      and (p_family is null
            or nullif(events.payload ->> 'family', '') = p_family)
      and (p_question_type is null
            or nullif(events.payload ->> 'question_type', '') = p_question_type)
      and (p_question_id is null or events.question_id = p_question_id)
      and (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_category is null or events.payload ->> 'category' = p_category)
      and (p_from is null or p_to is null or p_from < p_to)
)
select
    filtered.event_id,
    filtered.occurred_at,
    filtered.received_at,
    filtered.session_id,
    filtered.attempt_id,
    filtered.question_id,
    filtered.question_version,
    filtered.release_id,
    filtered.family,
    filtered.question_type,
    filtered.category,
    filtered.comment_text
from filtered
order by filtered.occurred_at desc, filtered.event_id;
$function$;


revoke all on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text
) from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz
) to service_role;
grant execute on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz
) to service_role;
grant execute on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz
) to service_role;
grant execute on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text
) to service_role;

comment on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected REPORT-001 summary over a consistently filtered event set.';
comment on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected REPORT-001 attempt/session rows with known-answer accuracy.';
comment on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected REPORT-001 question rows with known-answer accuracy.';
comment on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text
) is 'Protected REPORT-001 categorized comment rows.';

commit;
