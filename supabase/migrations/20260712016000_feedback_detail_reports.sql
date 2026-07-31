begin;

-- REPORT-002 detail surfaces.  These functions deliberately project only
-- registry-matched events and use canonical question dimensions from the
-- authoritative registry view.  Client claims remain diagnostic fields only.
-- Recreate the 15000 status function in this transaction so a hosted caller
-- can prove that both detail RPCs, not only the registry cutover, are present.
drop function public.feedback_report_authority_status();

create function public.feedback_report_answers(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null
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
    p_to timestamptz default null
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
  and (p_from is null or events.occurred_at >= p_from)
  and (p_to is null or events.occurred_at < p_to)
  and (p_from is null or p_to is null or p_from < p_to)
order by events.occurred_at desc, events.event_id;
$function$;

create function public.feedback_report_authority_status()
returns table (
    authority_revision text,
    business_reports_authoritative boolean,
    registered_release_count bigint,
    registered_question_count bigint,
    registered_choice_count bigint,
    detail_revision text,
    detail_reports_authoritative boolean
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
    ) as registered_choice_count,
    'detail_v1'::text as detail_revision,
    true as detail_reports_authoritative;
$function$;

revoke all on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz
) from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_authority_status()
from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz
) to service_role;
grant execute on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz
) to service_role;
grant execute on function public.feedback_report_authority_status()
to service_role;

comment on function public.feedback_report_answers(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected answer-event detail with registry-derived identity and correctness.';
comment on function public.feedback_report_proposals(
    text, text, text, text, timestamptz, timestamptz
) is 'Protected proposed/rejected setting detail attributed only by registered question membership.';
comment on function public.feedback_report_authority_status() is
    'Single-row registry_v1/detail_v1 proof that authoritative business and detail reports are installed.';

commit;
