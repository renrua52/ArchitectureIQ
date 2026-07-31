begin;

-- STATS-004 serves all six business-report pages from one PostgreSQL
-- statement snapshot.  The Edge Function must call only this RPC for a UI
-- refresh; splitting these reads across RPCs would create independent MVCC
-- snapshots even when every request carried the same timestamp.
--
-- pages_json is text, rather than jsonb, at the RPC boundary.  PostgREST
-- serializes the canonical JSON once and the Edge JavaScript runtime forwards
-- the string without parsing its integer-valued fields as IEEE-754 numbers.
create function public.feedback_report_business_snapshot(
    p_release_id text default null,
    p_family text default null,
    p_question_type text default null,
    p_question_id text default null,
    p_from timestamptz default null,
    p_to timestamptz default null,
    p_limit integer default 200
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
            parameters.to_at
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
            parameters.to_at
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
            parameters.to_at
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
            parameters.to_at
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
            parameters.to_at
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
            null
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

revoke all on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer
)
from public, anon, authenticated, service_role;

grant execute on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer
)
to service_role;

comment on function public.feedback_report_business_snapshot(
    text, text, text, text, timestamptz, timestamptz, integer
) is 'Protected STATS-004 single-statement snapshot of all six authoritative business report pages.';

commit;
