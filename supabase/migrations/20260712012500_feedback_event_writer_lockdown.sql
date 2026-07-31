begin;

-- Rollout order matters: apply the preceding RPC migration, deploy and verify
-- the RPC-aware feedback-ingest Edge Function, then apply this lockdown.  Old
-- Edge instances that remain briefly in flight fail closed and can be retried.
-- SELECT remains available for the protected report RPCs; only direct INSERT
-- is removed so every application write participates in advisory locking and
-- exact logical-event conflict classification.
revoke insert on table public.feedback_events from service_role;
grant select on table public.feedback_events to service_role;

commit;
