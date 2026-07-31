begin;

-- STATS-003 cutover: preserve the protected REPORT-001 function signatures and
-- return-column order while replacing every browser-reported dimension and
-- correctness value with the immutable registry projection from 14000.
drop function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz
);
drop function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz
);
drop function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz
);
drop function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text
);

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
  and (p_from is null or events.occurred_at >= p_from)
  and (p_to is null or events.occurred_at < p_to)
  and (p_category is null or events.payload ->> 'category' = p_category)
  and (p_from is null or p_to is null or p_from < p_to)
order by events.occurred_at desc, events.event_id;
$function$;

-- A hosted verifier needs positive evidence that this atomic cutover, rather
-- than only the additive registry migration, has been applied.  Keeping the
-- revision marker in the same transaction as the four replaced business RPCs
-- makes a successful status query an explicit 15000 deployment proof.
create function public.feedback_report_authority_status()
returns table (
    authority_revision text,
    business_reports_authoritative boolean,
    registered_release_count bigint,
    registered_question_count bigint,
    registered_choice_count bigint
)
language sql
stable
security invoker
set search_path = ''
as $function$
select
    'registry_v1'::text as authority_revision,
    true as business_reports_authoritative,
    (
        select pg_catalog.count(*)
        from public.feedback_quiz_releases
    ) as registered_release_count,
    (
        select pg_catalog.count(*)
        from public.feedback_quiz_questions
    ) as registered_question_count,
    (
        select pg_catalog.count(*)
        from public.feedback_quiz_choices
    ) as registered_choice_count;
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
revoke all on function public.feedback_report_authority_status()
from public, anon, authenticated, service_role;

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
grant execute on function public.feedback_report_authority_status()
to service_role;

comment on function public.feedback_report_summary(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected authoritative summary over registry-matched feedback events.';
comment on function public.feedback_report_sessions(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected authoritative session rows over registry-matched feedback events.';
comment on function public.feedback_report_questions(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected authoritative question rows with registry-derived correctness.';
comment on function public.feedback_report_comments(
    text, text, text, text, timestamptz, timestamptz, text
) is 'Protected categorized comments attributed only by registered question membership.';
comment on function public.feedback_report_authority_status() is
    'Single-row registry_v1 proof that the authoritative business-report cutover is installed.';

commit;
