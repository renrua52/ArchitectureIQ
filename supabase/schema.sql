-- ArchitectureIQ quiz: user system + session recordings.
-- Apply once as a DB owner (SQL Editor or direct connection).
--
-- Security model:
--   * All tables are RLS-enabled with ZERO policies and no anon grants,
--     so the browser-visible anon key cannot read or write any table.
--   * The only browser surface is three SECURITY DEFINER functions
--     (quiz_register / quiz_ingest_chunk / quiz_upsert_session), which run
--     as the owner and therefore bypass RLS.
--   * Group access passwords live in quiz_groups only (one row per group;
--     adding a group = one INSERT, no function edits).
--   * Admins read data via a direct Postgres connection (service role).

-- ---------------------------------------------------------------- tables

create table if not exists public.quiz_groups (
  group_name text primary key,
  password text not null unique
);

-- Seed the original group. Adding another group is one INSERT here:
--   insert into public.quiz_groups (group_name, password)
--   values ('group-b', 'its-password')
--   on conflict (group_name) do nothing;
insert into public.quiz_groups (group_name, password)
values ('main', 'tccrzrgsy')
on conflict (group_name) do nothing;

create table if not exists public.quiz_users (
  id uuid primary key default gen_random_uuid(),
  username text not null unique,
  token uuid not null default gen_random_uuid(),
  group_name text not null default 'main'
    references public.quiz_groups (group_name),
  created_at timestamptz not null default now()
);

-- Deployments created before groups existed: add the column (existing
-- users land in 'main', which was seeded above).
alter table public.quiz_users
  add column if not exists group_name text not null default 'main'
    references public.quiz_groups (group_name);

create table if not exists public.quiz_sessions (
  session_id text primary key,
  user_id uuid not null references public.quiz_users (id),
  pack text,
  score_correct integer not null default 0,
  score_total integer not null default 0,
  started_at timestamptz not null default now(),
  last_seen timestamptz not null default now(),
  meta jsonb not null default '{}'::jsonb
);

create table if not exists public.recording_chunks (
  id bigserial primary key,
  session_id text not null references public.quiz_sessions (session_id) on delete cascade,
  seq integer not null,
  events jsonb not null,
  created_at timestamptz not null default now(),
  unique (session_id, seq)
);

create index if not exists idx_recording_chunks_session
  on public.recording_chunks (session_id);
create index if not exists idx_quiz_sessions_user
  on public.quiz_sessions (user_id);

-- ------------------------------------------------------------ RLS lockdown

alter table public.quiz_users enable row level security;
alter table public.quiz_groups enable row level security;
alter table public.quiz_sessions enable row level security;
alter table public.recording_chunks enable row level security;

revoke all on public.quiz_users from anon, authenticated;
revoke all on public.quiz_groups from anon, authenticated;
revoke all on public.quiz_sessions from anon, authenticated;
revoke all on public.recording_chunks from anon, authenticated;

-- ------------------------------------------------------------------- RPCs
-- register / log in with a self-chosen username and the shared password.
-- Returns the per-user upload token. Existing username => same account back
-- (the password is shared, so "logging in as" an existing name is by design).

create or replace function public.quiz_register(p_username text, p_password text)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  v_name text := btrim(coalesce(p_username, ''));
  v_group text;
  v_user public.quiz_users%rowtype;
begin
  -- Password selects the group: each row in quiz_groups admits its own
  -- cohort, and the matched group is stamped on the user record.
  select group_name into v_group
  from public.quiz_groups
  where password = p_password;
  if v_group is null then
    raise exception 'invalid_password';
  end if;
  if char_length(v_name) < 1 or char_length(v_name) > 40 then
    raise exception 'invalid_username_length';
  end if;

  select * into v_user from public.quiz_users where username = v_name;
  if found then
    return json_build_object(
      'user_id', v_user.id,
      'username', v_user.username,
      'token', v_user.token,
      'group', v_user.group_name,
      'existed', true
    );
  end if;

  insert into public.quiz_users (username, group_name)
  values (v_name, v_group)
  returning * into v_user;

  return json_build_object(
    'user_id', v_user.id,
    'username', v_user.username,
    'token', v_user.token,
    'group', v_user.group_name,
    'existed', false
  );
end;
$$;

-- upsert one session's score / pack / meta. Caller must own the session.

create or replace function public.quiz_upsert_session(
  p_token uuid,
  p_session_id text,
  p_pack text,
  p_score_correct integer,
  p_score_total integer,
  p_meta jsonb default '{}'::jsonb
)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid;
  v_owner uuid;
begin
  select id into v_user_id from public.quiz_users where token = p_token;
  if v_user_id is null then
    raise exception 'invalid_token';
  end if;
  if p_session_id is null or char_length(p_session_id) > 80 then
    raise exception 'invalid_session_id';
  end if;

  select user_id into v_owner from public.quiz_sessions
  where session_id = p_session_id;
  if found and v_owner <> v_user_id then
    raise exception 'session_owner_mismatch';
  end if;

  insert into public.quiz_sessions
    (session_id, user_id, pack, score_correct, score_total, meta, last_seen)
  values
    (p_session_id, v_user_id, p_pack,
     greatest(0, coalesce(p_score_correct, 0)),
     greatest(0, coalesce(p_score_total, 0)),
     coalesce(p_meta, '{}'::jsonb), now())
  on conflict (session_id) do update set
    pack = excluded.pack,
    score_correct = excluded.score_correct,
    score_total = excluded.score_total,
    meta = public.quiz_sessions.meta || excluded.meta,
    last_seen = now();

  return json_build_object('ok', true);
end;
$$;

-- append one recording chunk. Idempotent per (session_id, seq).

create or replace function public.quiz_ingest_chunk(
  p_token uuid,
  p_session_id text,
  p_seq integer,
  p_events jsonb
)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid;
begin
  select id into v_user_id from public.quiz_users where token = p_token;
  if v_user_id is null then
    raise exception 'invalid_token';
  end if;
  if p_seq is null or p_seq < 0 or p_seq > 100000 then
    raise exception 'invalid_seq';
  end if;
  if p_events is null or jsonb_typeof(p_events) <> 'array'
     or jsonb_array_length(p_events) > 2000 then
    raise exception 'invalid_events';
  end if;

  -- chunk may arrive before the first score upsert: create the shell.
  insert into public.quiz_sessions (session_id, user_id, last_seen)
  values (p_session_id, v_user_id, now())
  on conflict (session_id) do update set last_seen = now();

  if (select user_id from public.quiz_sessions where session_id = p_session_id) <> v_user_id then
    raise exception 'session_owner_mismatch';
  end if;

  insert into public.recording_chunks (session_id, seq, events)
  values (p_session_id, p_seq, p_events)
  on conflict (session_id, seq) do update set events = excluded.events;

  return json_build_object('ok', true);
end;
$$;

-- Only these three functions are callable by the browser.

revoke all on function public.quiz_register(text, text) from public;
revoke all on function public.quiz_upsert_session(uuid, text, text, integer, integer, jsonb) from public;
revoke all on function public.quiz_ingest_chunk(uuid, text, integer, jsonb) from public;
grant execute on function public.quiz_register(text, text) to anon;
grant execute on function public.quiz_upsert_session(uuid, text, text, integer, integer, jsonb) to anon;
grant execute on function public.quiz_ingest_chunk(uuid, text, integer, jsonb) to anon;

-- ------------------------------------------------- per-user answer records
-- Authoritative per-user answer log: one row per (user, question).
-- quiz_list_answers lets the signed-in user resume where they left off
-- (refresh keeps their locked answers); quiz_record_answer is insert-only
-- per (user, question): the FIRST answer is permanent and can never be
-- overwritten by a redo. A re-answer only bumps the attempts/last_answered_at
-- audit fields (visible for proctoring) and returns the original answer.

create table if not exists public.quiz_answers (
  user_id uuid not null references public.quiz_users (id),
  question_id text not null,
  pack text,
  picked text not null,
  correct boolean not null,
  attempts integer not null default 1,
  first_answered_at timestamptz not null default now(),
  last_answered_at timestamptz not null default now(),
  primary key (user_id, question_id)
);

create index if not exists idx_quiz_answers_user_pack
  on public.quiz_answers (user_id, pack);

alter table public.quiz_answers enable row level security;
revoke all on public.quiz_answers from anon, authenticated;

create or replace function public.quiz_record_answer(
  p_token uuid,
  p_question_id text,
  p_picked text,
  p_correct boolean,
  p_pack text default null
)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid;
  v_picked text;
  v_correct boolean;
begin
  select id into v_user_id from public.quiz_users where token = p_token;
  if v_user_id is null then
    raise exception 'invalid_token';
  end if;
  if p_question_id is null or char_length(p_question_id) > 80
     or p_picked is null or char_length(p_picked) > 4 then
    raise exception 'invalid_answer';
  end if;

  -- First answer wins: insert, or (if this user already answered this
  -- question) keep the original row untouched except for the audit fields.
  insert into public.quiz_answers
    (user_id, question_id, pack, picked, correct, attempts)
  values
    (v_user_id, p_question_id, p_pack, p_picked, p_correct, 1)
  on conflict (user_id, question_id) do update set
    attempts = public.quiz_answers.attempts + 1,
    last_answered_at = now()
  returning picked, correct into v_picked, v_correct;

  return json_build_object(
    'duplicate', v_picked is distinct from p_picked
                 or v_correct is distinct from p_correct,
    'picked', v_picked,
    'correct', v_correct
  );
end;
$$;

create or replace function public.quiz_list_answers(p_token uuid, p_pack text default null)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid;
  v_rows json;
begin
  select id into v_user_id from public.quiz_users where token = p_token;
  if v_user_id is null then
    raise exception 'invalid_token';
  end if;

  select coalesce(json_agg(json_build_object(
           'question_id', a.question_id,
           'picked', a.picked,
           'correct', a.correct,
           'attempts', a.attempts
         ) order by a.first_answered_at), '[]'::json)
  into v_rows
  from public.quiz_answers a
  where a.user_id = v_user_id
    and (p_pack is null or a.pack = p_pack);

  return v_rows;
end;
$$;

revoke all on function public.quiz_record_answer(uuid, text, text, boolean, text) from public;
revoke all on function public.quiz_list_answers(uuid, text) from public;
grant execute on function public.quiz_record_answer(uuid, text, text, boolean, text) to anon;
grant execute on function public.quiz_list_answers(uuid, text) to anon;
