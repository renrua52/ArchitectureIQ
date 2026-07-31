begin;

-- STATS-003 stores immutable, publisher-attested quiz facts separately from
-- browser-supplied feedback payloads.  The ingestion service role may read
-- these tables for reporting, but cannot register or mutate releases.
create table public.feedback_quiz_releases (
    release_id text primary key,
    registry_schema_version text not null,
    manifest_sha256 text not null,
    registry_id text not null unique,
    question_count integer not null,
    choice_count integer not null,
    registered_at timestamptz not null default pg_catalog.statement_timestamp(),

    constraint feedback_quiz_releases_schema_check
        check (registry_schema_version = '1.0'),
    constraint feedback_quiz_releases_release_id_check
        check (release_id ~ '^release_[0-9a-f]{64}$'),
    constraint feedback_quiz_releases_manifest_sha256_check
        check (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    constraint feedback_quiz_releases_registry_id_check
        check (registry_id ~ '^registry_[0-9a-f]{64}$'),
    constraint feedback_quiz_releases_question_count_check
        check (question_count > 0),
    constraint feedback_quiz_releases_choice_count_check
        check (choice_count >= question_count)
);

create table public.feedback_quiz_questions (
    release_id text not null,
    question_id text not null,
    question_version text not null,
    family text not null,
    dataset_id text not null,
    question_type text not null,
    correct_letter text not null,
    correct_candidate_id text not null,
    choice_count integer not null,

    constraint feedback_quiz_questions_pkey
        primary key (release_id, question_id, question_version),
    constraint feedback_quiz_questions_release_question_key
        unique (release_id, question_id),
    constraint feedback_quiz_questions_release_fkey
        foreign key (release_id)
        references public.feedback_quiz_releases (release_id),
    constraint feedback_quiz_questions_version_check
        check (question_version ~ '^qv1_[0-9a-f]{64}$'),
    constraint feedback_quiz_questions_correct_letter_check
        check (correct_letter ~ '^[A-Z]$'),
    constraint feedback_quiz_questions_choice_count_check
        check (choice_count between 2 and 26),
    constraint feedback_quiz_questions_identifiers_check
        check (
            question_id = pg_catalog.btrim(question_id)
            and pg_catalog.length(question_id) between 1 and 200
            and question_id !~ E'[\r\n]'
            and family = pg_catalog.btrim(family)
            and pg_catalog.length(family) between 1 and 200
            and family !~ E'[\r\n]'
            and dataset_id = pg_catalog.btrim(dataset_id)
            and pg_catalog.length(dataset_id) between 1 and 200
            and dataset_id !~ E'[\r\n]'
            and question_type = pg_catalog.btrim(question_type)
            and pg_catalog.length(question_type) between 1 and 200
            and question_type !~ E'[\r\n]'
            and correct_candidate_id = pg_catalog.btrim(correct_candidate_id)
            and pg_catalog.length(correct_candidate_id) between 1 and 200
            and correct_candidate_id !~ E'[\r\n]'
        )
);

create table public.feedback_quiz_choices (
    release_id text not null,
    question_id text not null,
    question_version text not null,
    letter text not null,
    candidate_id text not null,

    constraint feedback_quiz_choices_pkey
        primary key (release_id, question_id, question_version, letter),
    constraint feedback_quiz_choices_candidate_key
        unique (release_id, question_id, question_version, candidate_id),
    constraint feedback_quiz_choices_letter_candidate_key
        unique (
            release_id,
            question_id,
            question_version,
            letter,
            candidate_id
        ),
    constraint feedback_quiz_choices_question_fkey
        foreign key (release_id, question_id, question_version)
        references public.feedback_quiz_questions (
            release_id,
            question_id,
            question_version
        ),
    constraint feedback_quiz_choices_letter_check
        check (letter ~ '^[A-Z]$'),
    constraint feedback_quiz_choices_candidate_id_check
        check (
            candidate_id = pg_catalog.btrim(candidate_id)
            and pg_catalog.length(candidate_id) between 1 and 200
            and candidate_id !~ E'[\r\n]'
        )
);

-- This deferred reference permits a data migration to insert questions before
-- choices while still proving that every declared answer is an actual choice.
alter table public.feedback_quiz_questions
add constraint feedback_quiz_questions_correct_choice_fkey
foreign key (
    release_id,
    question_id,
    question_version,
    correct_letter,
    correct_candidate_id
)
references public.feedback_quiz_choices (
    release_id,
    question_id,
    question_version,
    letter,
    candidate_id
)
deferrable initially deferred;

comment on table public.feedback_quiz_releases is
    'Immutable releases exported from a fully attested ArchitectureIQ quiz bundle.';
comment on table public.feedback_quiz_questions is
    'Authoritative question identity, dimensions, and correct choice per release.';
comment on table public.feedback_quiz_choices is
    'Authoritative letter-to-candidate mapping for one registered question version.';

create function public.reject_feedback_quiz_registry_mutation()
returns trigger
language plpgsql
set search_path = ''
as $function$
begin
    raise exception using
        errcode = '55000',
        message = 'feedback quiz registry is append-only';
end;
$function$;

revoke all on function public.reject_feedback_quiz_registry_mutation()
from public, anon, authenticated, service_role;

create trigger feedback_quiz_releases_append_only
before update or delete or truncate on public.feedback_quiz_releases
for each statement execute function
    public.reject_feedback_quiz_registry_mutation();

create trigger feedback_quiz_questions_append_only
before update or delete or truncate on public.feedback_quiz_questions
for each statement execute function
    public.reject_feedback_quiz_registry_mutation();

create trigger feedback_quiz_choices_append_only
before update or delete or truncate on public.feedback_quiz_choices
for each statement execute function
    public.reject_feedback_quiz_registry_mutation();

-- Deferred cardinality checks make one reviewed data migration the atomic
-- registration boundary.  A release cannot commit with a partial question or
-- choice inventory, even if the inserts themselves are syntactically valid.
create function public.lock_feedback_quiz_question_version()
returns trigger
language plpgsql
set search_path = ''
as $function$
begin
    perform pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            pg_catalog.concat(
                'architecture_iq.feedback_quiz_question_version:',
                new.question_version
            ),
            0
        )
    );
    return new;
end;
$function$;

revoke all on function public.lock_feedback_quiz_question_version()
from public, anon, authenticated, service_role;

create trigger feedback_quiz_question_version_lock
before insert on public.feedback_quiz_questions
for each row execute function
    public.lock_feedback_quiz_question_version();

create function public.validate_feedback_quiz_question_inventory()
returns trigger
language plpgsql
set search_path = ''
as $function$
declare
    v_choice_count bigint;
    v_question public.feedback_quiz_questions%rowtype;
begin
    select questions.*
    into v_question
    from public.feedback_quiz_questions as questions
    where questions.release_id = new.release_id
      and questions.question_id = new.question_id
      and questions.question_version = new.question_version;

    if not found then
        raise exception using
            errcode = '23503',
            message = 'registered choice has no parent question';
    end if;

    select pg_catalog.count(*)
    into v_choice_count
    from public.feedback_quiz_choices as choices
    where choices.release_id = v_question.release_id
      and choices.question_id = v_question.question_id
      and choices.question_version = v_question.question_version;

    if v_choice_count <> v_question.choice_count then
        raise exception using
            errcode = '23514',
            message = 'registered question choice_count does not match choices';
    end if;

    if exists (
        select 1
        from public.feedback_quiz_questions as other
        where other.question_version = v_question.question_version
          and (
            other.question_id,
            other.family,
            other.dataset_id,
            other.question_type,
            other.correct_letter,
            other.correct_candidate_id,
            other.choice_count
          ) is distinct from (
            v_question.question_id,
            v_question.family,
            v_question.dataset_id,
            v_question.question_type,
            v_question.correct_letter,
            v_question.correct_candidate_id,
            v_question.choice_count
          )
    ) then
        raise exception using
            errcode = '23514',
            message = 'question_version metadata differs across releases';
    end if;

    if exists (
        select 1
        from public.feedback_quiz_questions as other
        where other.question_version = v_question.question_version
          and other.release_id <> v_question.release_id
          and (
            exists (
                select current_choices.letter, current_choices.candidate_id
                from public.feedback_quiz_choices as current_choices
                where current_choices.release_id = v_question.release_id
                  and current_choices.question_id = v_question.question_id
                  and current_choices.question_version = v_question.question_version
                except
                select other_choices.letter, other_choices.candidate_id
                from public.feedback_quiz_choices as other_choices
                where other_choices.release_id = other.release_id
                  and other_choices.question_id = other.question_id
                  and other_choices.question_version = other.question_version
            )
            or exists (
                select other_choices.letter, other_choices.candidate_id
                from public.feedback_quiz_choices as other_choices
                where other_choices.release_id = other.release_id
                  and other_choices.question_id = other.question_id
                  and other_choices.question_version = other.question_version
                except
                select current_choices.letter, current_choices.candidate_id
                from public.feedback_quiz_choices as current_choices
                where current_choices.release_id = v_question.release_id
                  and current_choices.question_id = v_question.question_id
                  and current_choices.question_version = v_question.question_version
            )
          )
    ) then
        raise exception using
            errcode = '23514',
            message = 'question_version choices differ across releases';
    end if;
    return null;
end;
$function$;

create function public.validate_feedback_quiz_release_inventory()
returns trigger
language plpgsql
set search_path = ''
as $function$
declare
    v_question_count bigint;
    v_choice_count bigint;
    v_expected_question_count integer;
    v_expected_choice_count integer;
begin
    select releases.question_count, releases.choice_count
    into v_expected_question_count, v_expected_choice_count
    from public.feedback_quiz_releases as releases
    where releases.release_id = new.release_id;

    if not found then
        raise exception using
            errcode = '23503',
            message = 'registered question or choice has no parent release';
    end if;

    select pg_catalog.count(*)
    into v_question_count
    from public.feedback_quiz_questions as questions
    where questions.release_id = new.release_id;

    select pg_catalog.count(*)
    into v_choice_count
    from public.feedback_quiz_choices as choices
    where choices.release_id = new.release_id;

    if v_question_count <> v_expected_question_count then
        raise exception using
            errcode = '23514',
            message = 'registered release question_count does not match questions';
    end if;
    if v_choice_count <> v_expected_choice_count then
        raise exception using
            errcode = '23514',
            message = 'registered release choice_count does not match choices';
    end if;
    return null;
end;
$function$;

revoke all on function public.validate_feedback_quiz_question_inventory()
from public, anon, authenticated, service_role;
revoke all on function public.validate_feedback_quiz_release_inventory()
from public, anon, authenticated, service_role;

create constraint trigger feedback_quiz_question_inventory_complete
after insert on public.feedback_quiz_questions
deferrable initially deferred
for each row execute function
    public.validate_feedback_quiz_question_inventory();

create constraint trigger feedback_quiz_choice_inventory_complete
after insert on public.feedback_quiz_choices
deferrable initially deferred
for each row execute function
    public.validate_feedback_quiz_question_inventory();

create constraint trigger feedback_quiz_release_inventory_complete
after insert on public.feedback_quiz_releases
deferrable initially deferred
for each row execute function
    public.validate_feedback_quiz_release_inventory();

create constraint trigger feedback_quiz_question_release_inventory_complete
after insert on public.feedback_quiz_questions
deferrable initially deferred
for each row execute function
    public.validate_feedback_quiz_release_inventory();

create constraint trigger feedback_quiz_choice_release_inventory_complete
after insert on public.feedback_quiz_choices
deferrable initially deferred
for each row execute function
    public.validate_feedback_quiz_release_inventory();

alter table public.feedback_quiz_releases enable row level security;
alter table public.feedback_quiz_releases force row level security;
alter table public.feedback_quiz_questions enable row level security;
alter table public.feedback_quiz_questions force row level security;
alter table public.feedback_quiz_choices enable row level security;
alter table public.feedback_quiz_choices force row level security;

revoke all on table
    public.feedback_quiz_releases,
    public.feedback_quiz_questions,
    public.feedback_quiz_choices
from public, anon, authenticated, service_role;

grant select on table
    public.feedback_quiz_releases,
    public.feedback_quiz_questions,
    public.feedback_quiz_choices
to service_role;

-- Dynamic projection deliberately permits later release registration to make
-- older raw events resolvable.  It never mutates feedback_events and never
-- guesses a release from question_id/question_version alone.
create view public.feedback_authoritative_events
with (security_invoker = true, security_barrier = true)
as
with joined as (
    select
        events.*,
        nullif(events.payload ->> 'attempt_id', '') as report_attempt_id,
        nullif(events.payload ->> 'release_id', '') as claimed_release_id,
        nullif(events.payload ->> 'family', '') as claimed_family,
        nullif(events.payload ->> 'dataset_id', '') as claimed_dataset_id,
        nullif(events.payload ->> 'question_type', '') as claimed_question_type,
        releases.release_id as known_release_id,
        releases.registry_id as authoritative_registry_id,
        questions.question_id as authoritative_question_id,
        questions.question_version as authoritative_question_version,
        questions.release_id as authoritative_release_id,
        questions.family as authoritative_family,
        questions.dataset_id as authoritative_dataset_id,
        questions.question_type as authoritative_question_type,
        questions.correct_letter as authoritative_correct_letter,
        choices.letter as authoritative_selected_letter,
        choices.candidate_id as authoritative_selected_candidate_id
    from public.feedback_events as events
    left join public.feedback_quiz_releases as releases
        on releases.release_id = nullif(events.payload ->> 'release_id', '')
    left join public.feedback_quiz_questions as questions
        on questions.release_id = releases.release_id
       and questions.question_id = events.question_id
       and questions.question_version = events.question_version
    left join public.feedback_quiz_choices as choices
        on choices.release_id = questions.release_id
       and choices.question_id = questions.question_id
       and choices.question_version = questions.question_version
       and choices.letter = nullif(events.payload ->> 'selected_letter', '')
), resolved as (
    select
        joined.*,
        case
            when joined.claimed_release_id is null then 'missing_release'
            when joined.known_release_id is null then 'unknown_release'
            when joined.authoritative_question_id is null
                then 'question_not_in_release'
            else 'matched'
        end as registry_status,
        case
            when joined.event_type <> 'answer_submitted' then 'not_answer'
            when joined.authoritative_question_id is null
                then 'unresolved_registry'
            when joined.authoritative_selected_letter is null
                then 'invalid_selected_letter'
            when nullif(
                joined.payload ->> 'selected_candidate_id',
                ''
            ) is not null
             and nullif(
                joined.payload ->> 'selected_candidate_id',
                ''
             ) <> joined.authoritative_selected_candidate_id
                then 'selected_candidate_mismatch'
            else 'resolved'
        end as answer_status,
        case
            when joined.event_type = 'answer_submitted'
             and joined.authoritative_question_id is not null
             and joined.authoritative_selected_letter is not null
             and (
                nullif(joined.payload ->> 'selected_candidate_id', '') is null
                or nullif(joined.payload ->> 'selected_candidate_id', '')
                    = joined.authoritative_selected_candidate_id
             )
                then joined.authoritative_selected_letter
                    = joined.authoritative_correct_letter
            else null
        end as authoritative_is_correct
    from joined
)
select
    resolved.*,
    resolved.authoritative_question_id is not null
        and (
            (
                resolved.claimed_family is not null
                and resolved.claimed_family <> resolved.authoritative_family
            )
            or (
                resolved.claimed_dataset_id is not null
                and resolved.claimed_dataset_id
                    <> resolved.authoritative_dataset_id
            )
            or (
                resolved.claimed_question_type is not null
                and resolved.claimed_question_type
                    <> resolved.authoritative_question_type
            )
        ) as client_context_mismatch,
    coalesce(
        resolved.authoritative_is_correct is not null
            and pg_catalog.jsonb_typeof(
                resolved.payload -> 'is_correct'
            ) = 'boolean'
            and (resolved.payload -> 'is_correct' = 'true'::jsonb)
                <> resolved.authoritative_is_correct,
        false
    ) as client_correctness_mismatch
from resolved;

revoke all on table public.feedback_authoritative_events
from public, anon, authenticated, service_role;
grant select on table public.feedback_authoritative_events to service_role;

-- Additive quality surface: business report schemas stay stable while this row
-- makes unresolved registry claims and client/server disagreements visible.
create function public.feedback_report_registry_quality(
    p_from timestamptz default null,
    p_to timestamptz default null
)
returns table (
    registered_release_count bigint,
    registered_question_count bigint,
    registered_choice_count bigint,
    registry_available boolean,
    raw_event_count bigint,
    authoritative_event_count bigint,
    excluded_event_count bigint,
    missing_release_event_count bigint,
    unknown_release_event_count bigint,
    question_not_in_release_event_count bigint,
    raw_answer_count bigint,
    authoritative_answer_count bigint,
    unresolved_answer_count bigint,
    invalid_selected_letter_answer_count bigint,
    selected_candidate_mismatch_answer_count bigint,
    unmatched_comment_count bigint,
    unmatched_proposal_count bigint,
    client_context_mismatch_event_count bigint,
    client_correctness_mismatch_answer_count bigint,
    registry_match_rate numeric,
    answer_resolution_rate numeric
)
language sql
stable
security invoker
set search_path = ''
as $function$
with registry as (
    select
        (select pg_catalog.count(*)
            from public.feedback_quiz_releases) as release_count,
        (select pg_catalog.count(*)
            from public.feedback_quiz_questions) as question_count,
        (select pg_catalog.count(*)
            from public.feedback_quiz_choices) as choice_count
), filtered as (
    select events.*
    from public.feedback_authoritative_events as events
    where (p_from is null or events.occurred_at >= p_from)
      and (p_to is null or events.occurred_at < p_to)
      and (p_from is null or p_to is null or p_from < p_to)
), metrics as (
    select
        pg_catalog.count(*) as raw_event_count,
        pg_catalog.count(*) filter (
            where registry_status = 'matched'
        ) as authoritative_event_count,
        pg_catalog.count(*) filter (
            where registry_status <> 'matched'
        ) as excluded_event_count,
        pg_catalog.count(*) filter (
            where registry_status = 'missing_release'
        ) as missing_release_event_count,
        pg_catalog.count(*) filter (
            where registry_status = 'unknown_release'
        ) as unknown_release_event_count,
        pg_catalog.count(*) filter (
            where registry_status = 'question_not_in_release'
        ) as question_not_in_release_event_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
        ) as raw_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct is not null
        ) as authoritative_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and authoritative_is_correct is null
        ) as unresolved_answer_count,
        pg_catalog.count(*) filter (
            where answer_status = 'invalid_selected_letter'
        ) as invalid_selected_letter_answer_count,
        pg_catalog.count(*) filter (
            where answer_status = 'selected_candidate_mismatch'
        ) as selected_candidate_mismatch_answer_count,
        pg_catalog.count(*) filter (
            where event_type = 'comment_submitted'
              and registry_status <> 'matched'
        ) as unmatched_comment_count,
        pg_catalog.count(*) filter (
            where event_type in (
                'custom_setting_proposed',
                'custom_setting_rejected'
            )
              and registry_status <> 'matched'
        ) as unmatched_proposal_count,
        pg_catalog.count(*) filter (
            where client_context_mismatch
        ) as client_context_mismatch_event_count,
        pg_catalog.count(*) filter (
            where event_type = 'answer_submitted'
              and client_correctness_mismatch
        ) as client_correctness_mismatch_answer_count
    from filtered
)
select
    registry.release_count,
    registry.question_count,
    registry.choice_count,
    registry.release_count > 0,
    metrics.raw_event_count,
    metrics.authoritative_event_count,
    metrics.excluded_event_count,
    metrics.missing_release_event_count,
    metrics.unknown_release_event_count,
    metrics.question_not_in_release_event_count,
    metrics.raw_answer_count,
    metrics.authoritative_answer_count,
    metrics.unresolved_answer_count,
    metrics.invalid_selected_letter_answer_count,
    metrics.selected_candidate_mismatch_answer_count,
    metrics.unmatched_comment_count,
    metrics.unmatched_proposal_count,
    metrics.client_context_mismatch_event_count,
    metrics.client_correctness_mismatch_answer_count,
    pg_catalog.round(
        metrics.authoritative_event_count::numeric
        / nullif(metrics.raw_event_count, 0),
        4
    ),
    pg_catalog.round(
        metrics.authoritative_answer_count::numeric
        / nullif(metrics.raw_answer_count, 0),
        4
    )
from registry
cross join metrics;
$function$;

create function public.feedback_report_event_resolution(
    p_event_id text
)
returns table (
    event_id text,
    event_type text,
    occurred_at timestamptz,
    received_at timestamptz,
    session_id text,
    attempt_id text,
    client_release_id text,
    registry_status text,
    answer_status text,
    registry_id text,
    release_id text,
    question_id text,
    question_version text,
    family text,
    dataset_id text,
    question_type text,
    selected_letter text,
    client_selected_candidate_id text,
    selected_candidate_id text,
    authoritative_is_correct boolean,
    client_is_correct boolean,
    client_context_mismatch boolean,
    client_correctness_mismatch boolean
)
language sql
stable
security invoker
set search_path = ''
as $function$
with matched as (
    select
        events.event_id,
        events.event_type,
        events.occurred_at,
        events.received_at,
        events.session_id,
        events.report_attempt_id as attempt_id,
        events.claimed_release_id as client_release_id,
        events.registry_status,
        events.answer_status,
        events.authoritative_registry_id as registry_id,
        events.authoritative_release_id as release_id,
        events.authoritative_question_id as question_id,
        events.authoritative_question_version as question_version,
        events.authoritative_family as family,
        events.authoritative_dataset_id as dataset_id,
        events.authoritative_question_type as question_type,
        nullif(events.payload ->> 'selected_letter', '') as selected_letter,
        nullif(
            events.payload ->> 'selected_candidate_id',
            ''
        ) as client_selected_candidate_id,
        events.authoritative_selected_candidate_id as selected_candidate_id,
        events.authoritative_is_correct,
        case
            when pg_catalog.jsonb_typeof(
                events.payload -> 'is_correct'
            ) = 'boolean'
                then events.payload -> 'is_correct' = 'true'::jsonb
            else null
        end as client_is_correct,
        events.client_context_mismatch,
        events.client_correctness_mismatch
    from public.feedback_authoritative_events as events
    where events.event_id = p_event_id
)
select matched.*
from matched
union all
select
    p_event_id,
    null::text,
    null::timestamptz,
    null::timestamptz,
    null::text,
    null::text,
    null::text,
    'not_found'::text,
    'not_found'::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::text,
    null::boolean,
    null::boolean,
    false,
    false
where not exists (select 1 from matched);
$function$;

revoke all on function public.feedback_report_registry_quality(
    timestamptz, timestamptz
)
from public, anon, authenticated, service_role;
revoke all on function public.feedback_report_event_resolution(text)
from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_registry_quality(
    timestamptz, timestamptz
)
to service_role;
grant execute on function public.feedback_report_event_resolution(text)
to service_role;

comment on function public.feedback_report_registry_quality(
    timestamptz, timestamptz
) is 'Protected STATS-003 registry coverage and mismatch counts over authoritative report filters.';
comment on function public.feedback_report_event_resolution(text) is
    'Protected exact-event registry resolution for operator and hosted-verifier evidence.';

commit;
