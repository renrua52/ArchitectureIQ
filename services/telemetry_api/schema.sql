-- Run in Supabase SQL Editor (or any Postgres).

create table if not exists quiz_events (
  id bigserial primary key,
  session_id text not null,
  event_type text not null,
  question_id text,
  payload jsonb not null default '{}'::jsonb,
  duration_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists idx_quiz_events_created on quiz_events (created_at);
create index if not exists idx_quiz_events_session on quiz_events (session_id);
create index if not exists idx_quiz_events_question on quiz_events (question_id);
