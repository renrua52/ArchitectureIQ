begin;

-- Harden the original service-role reporting views without changing any
-- existing column name, order, or type.  New answer-quality columns are
-- appended so existing SELECT lists and consumers remain compatible.

create or replace view public.feedback_session_summary
with (security_invoker = true, security_barrier = true)
as
select
    session_id,
    nullif(payload ->> 'attempt_id', '') as attempt_id,
    min(occurred_at) as started_at,
    max(occurred_at) as last_event_at,
    min(received_at) as first_received_at,
    max(received_at) as last_received_at,
    array_remove(
        array_agg(distinct nullif(payload ->> 'release_id', '')),
        null
    ) as release_ids,
    array_remove(
        array_agg(distinct nullif(payload ->> 'family', '')),
        null
    ) as families,
    array_remove(
        array_agg(distinct nullif(payload ->> 'question_type', '')),
        null
    ) as question_types,
    count(*) as event_count,
    count(distinct (question_id, question_version)) as question_count,
    count(*) filter (where event_type = 'answer_submitted') as answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and payload -> 'is_correct' = 'true'::jsonb
    ) as correct_answer_count,
    round(
        count(*) filter (
            where event_type = 'answer_submitted'
              and payload -> 'is_correct' = 'true'::jsonb
        )::numeric
        / nullif(
            count(*) filter (
                where event_type = 'answer_submitted'
                  and jsonb_typeof(payload -> 'is_correct') = 'boolean'
            ),
            0
        ),
        4
    ) as accuracy,
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
    count(*) filter (where event_type = 'comment_submitted') as comment_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and jsonb_typeof(payload -> 'is_correct') = 'boolean'
    ) as known_answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and payload -> 'is_correct' = 'false'::jsonb
    ) as incorrect_answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and jsonb_typeof(payload -> 'is_correct') is distinct from 'boolean'
    ) as unknown_answer_count
from public.feedback_events
group by session_id, nullif(payload ->> 'attempt_id', '');


create or replace view public.feedback_question_stats
with (security_invoker = true, security_barrier = true)
as
select
    question_id,
    question_version,
    nullif(payload ->> 'release_id', '') as release_id,
    nullif(payload ->> 'family', '') as family,
    nullif(payload ->> 'dataset_id', '') as dataset_id,
    nullif(payload ->> 'question_type', '') as question_type,
    min(occurred_at) as first_event_at,
    max(occurred_at) as last_event_at,
    count(distinct session_id) as session_count,
    count(distinct (
        session_id,
        coalesce(nullif(payload ->> 'attempt_id', ''), '')
    )) as attempt_count,
    count(*) filter (where event_type = 'answer_submitted') as answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and payload -> 'is_correct' = 'true'::jsonb
    ) as correct_answer_count,
    round(
        count(*) filter (
            where event_type = 'answer_submitted'
              and payload -> 'is_correct' = 'true'::jsonb
        )::numeric
        / nullif(
            count(*) filter (
                where event_type = 'answer_submitted'
                  and jsonb_typeof(payload -> 'is_correct') = 'boolean'
            ),
            0
        ),
        4
    ) as accuracy,
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
    count(*) filter (where event_type = 'comment_submitted') as comment_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and jsonb_typeof(payload -> 'is_correct') = 'boolean'
    ) as known_answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and payload -> 'is_correct' = 'false'::jsonb
    ) as incorrect_answer_count,
    count(*) filter (
        where event_type = 'answer_submitted'
          and jsonb_typeof(payload -> 'is_correct') is distinct from 'boolean'
    ) as unknown_answer_count
from public.feedback_events
group by
    question_id,
    question_version,
    nullif(payload ->> 'release_id', ''),
    nullif(payload ->> 'family', ''),
    nullif(payload ->> 'dataset_id', ''),
    nullif(payload ->> 'question_type', '');


-- Keep the legacy integer output columns, but never cast arbitrary JSON text or
-- an out-of-range/fractional JSON number to integer.  The outer CASE establishes
-- the JSON type before the inner numeric operations are evaluated.
create or replace view public.feedback_proposals
with (security_invoker = true, security_barrier = true)
as
select
    event_id,
    occurred_at,
    received_at,
    session_id,
    nullif(payload ->> 'attempt_id', '') as attempt_id,
    question_id,
    question_version,
    nullif(payload ->> 'release_id', '') as release_id,
    nullif(payload ->> 'family', '') as family,
    nullif(payload ->> 'question_type', '') as question_type,
    case event_type
        when 'custom_setting_proposed' then 'proposed'
        when 'custom_setting_rejected' then 'rejected'
    end as setting_status,
    payload ->> 'label' as label,
    payload -> 'setting' as setting,
    payload -> 'inherited_from' as inherited_from,
    case
        when jsonb_typeof(payload -> 'n_seeds') = 'number' then
            case
                when (payload ->> 'n_seeds')::numeric
                        = trunc((payload ->> 'n_seeds')::numeric)
                  and (payload ->> 'n_seeds')::numeric
                        between -2147483648 and 2147483647
                then ((payload ->> 'n_seeds')::numeric)::integer
                else null
            end
        else null
    end as n_seeds,
    case
        when jsonb_typeof(payload -> 'base_seed') = 'number' then
            case
                when (payload ->> 'base_seed')::numeric
                        = trunc((payload ->> 'base_seed')::numeric)
                  and (payload ->> 'base_seed')::numeric
                        between -2147483648 and 2147483647
                then ((payload ->> 'base_seed')::numeric)::integer
                else null
            end
        else null
    end as base_seed
from public.feedback_events
where event_type in ('custom_setting_proposed', 'custom_setting_rejected');


revoke all on table
    public.feedback_session_summary,
    public.feedback_question_stats,
    public.feedback_proposals
from public, anon, authenticated;

grant select on table
    public.feedback_session_summary,
    public.feedback_question_stats,
    public.feedback_proposals
to service_role;

comment on view public.feedback_session_summary is
    'Service-role raw session/attempt summary with question-version counts and known-answer accuracy.';
comment on view public.feedback_question_stats is
    'Service-role raw per-question context summary with session-scoped attempts and known-answer accuracy.';
comment on view public.feedback_proposals is
    'Service-role raw proposed/rejected setting rows; optional integer seed fields are range-safe and never make the view fail.';

commit;
