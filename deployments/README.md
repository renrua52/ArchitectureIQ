# ArchitectureIQ deployment ledger

`tools/deployment_ledger.py` maintains the retrospective audit trail for
QPUB-008:

```text
release → source commit → provider deploy/site/entrypoint → hosted evidence
```

This ledger is deliberately **not runtime self-attestation** and is also not a
provider-signed transparency log. A value displayed by the deployed application,
an operator statement, or a locally passing preflight does not prove which
source a provider deployed. Source mapping requires a separately captured raw
provider-control-plane export, a normalized mapping envelope, and a reviewer
distinct from the recorder. The ledger hash-binds and cross-checks that reviewed
capture; it does not manufacture provider proof or add a provider signature that
was not present.

Accordingly, the successful operational state is named `ACTIVATED_REVIEWED`,
not `PROVIDER_VERIFIED` or `ACTIVE_VERIFIED`. Cryptographic provider verification
would require a provider API response with verifiable provenance or signature,
which Streamlit Community Cloud does not currently supply to this repository.
The preceding complete-evidence state is likewise named
`READY_FOR_REVIEWED_ACTIVATION`; “ready” means ready for a maintainer decision,
not provider-verified readiness.

## Files

The intended tracked layout is:

```text
deployments/
├── README.md
├── ledger.jsonl
└── evidence/
    └── <deployment_key>/
        ├── preflight.json
        ├── postgres-acceptance.json
        ├── hosted-roundtrip.json
        ├── provider-export.json (or .png/.pdf/.txt)
        └── provider-mapping.json
```

`ledger.jsonl` need not exist before the first real event. `verify` and `list`
treat a missing ledger as an empty ledger and never create a placeholder record.
Evidence paths are repository-relative regular files. Symlinks, paths outside
the repository, missing evidence, changed bytes, and incorrect raw SHA-256
values fail closed.

Do not put DSNs, Bearer tokens, cookies, signed URLs, URL userinfo, Streamlit
secrets, or evidence containing credentials in this directory. The CLI prints
only safe record summaries and never echoes evidence bodies.

## Event sequence

The global JSONL chain may interleave independent deployments. Each
`deployment_key` has this state machine:

```text
candidate_attested
        │
        ▼
deployment_declared
        │
        ├── postgres_accepted ───────────────┐
        ├── roundtrip_accepted               ├── any order, exactly once
        └── source_mapping_attested ─────────┘
                         │
                         ▼
                    activated
                         │
                         ├── superseded
                         └── rolled_back
```

Activation requires all three evidence events and a reviewer distinct from the
recorder. Source-mapping and terminal events have the same two-person rule.
After activation, only one terminal event is allowed. Nothing may follow a
terminal event for that deployment key, and a superseding replacement must
already be activated and non-terminal. Several deployment keys may reference the
same immutable release, but provider/project/deploy identities, hosted evidence
hashes, raw provider-export hashes, and roundtrip run/event/request identities
may not be reused across deployment keys.

The events mean:

- `candidate_attested`: binds the canonical manifest and registry, their raw
  hashes and counts, a full Git commit, repo/branch/entrypoint, the `report-app`
  rollout fingerprint, and a local-static preflight. Manifest, registry, and
  entrypoint bytes are read from the declared Git commit. Every path listed by
  preflight is also reread from that commit and the rollout fingerprint is
  independently recomputed; dirty working-tree bytes and self-reported PASS
  fields are not sufficient. Git inspection uses a fixed minimal environment,
  ignores inherited `GIT_*` redirection, and disables replacement objects.
- `deployment_declared`: records environment, target label, provider, project,
  provider deploy ID, a normalized credential-free HTTPS site URL, backend
  project ID, and SHA-256 values of the credential-free ingest/report origins.
  This is still only a deployment declaration.
- `postgres_accepted`: requires contacted, rollback-confirmed, all-PASS staging
  acceptance whose release, registry, counts, authority revisions, and target
  match the candidate and declaration.
- `roundtrip_accepted`: requires the authoritative hosted envelope
  (`authority_mode=authoritative`) to match the manifest and registry and to
  demonstrate authority status, detail reports, business snapshot, successful
  batch, first write, conflict behavior, and the REPORT-002 identity-filter
  proof. In particular, `session_attempt_filters_verified` must be `true`,
  which means the real session/attempt pair returned the exact uploaded trace
  and both wrong-session and wrong-attempt controls returned empty six-view
  snapshots. The current verifier output does not contain endpoint origins;
  the ledger binds its unique run/request IDs to the reviewed deployment
  context and origin hashes, but this association
  remains an operator-reviewed claim rather than provider cryptographic proof.
- `source_mapping_attested`: accepts only an
  `architecture_iq_provider_deployment_mapping` envelope with
  `mapping_authority=reviewed_provider_control_plane_capture`, a separate
  hash-bound raw provider export, `deployment_status=ready`, deployment/capture
  timestamps, and a distinct reviewer. Its environment/context,
  provider/project/deploy/site, and repo/branch/commit/entrypoint/release/
  fingerprint fields must exactly match the preceding records. The event name
  means “maintainers attested this reviewed capture”; it does not mean the
  provider cryptographically signed it.
- `activated`, `superseded`, and `rolled_back`: reviewer decisions. They do not
  alter or delete earlier evidence.

Until `source_mapping_attested` exists, the honest state remains
`DEPLOYMENT_DECLARED_SOURCE_MAPPING_UNVERIFIED`, even when local, PostgreSQL, and
hosted behavior checks pass.
After all evidence and review, the state becomes `ACTIVATED_REVIEWED`. A hosted
roundtrip whose `verified_at` predates the provider-reported `deployed_at` cannot
be activated.

## Hash and encoding contract

Every ledger line is one exact canonical UTF-8 JSON object followed by one LF:

- `schema_version` is `1.0`;
- `record_type` is `architecture_iq_deployment_event`;
- object keys are sorted, separators are `,` and `:`, and non-finite numbers,
  duplicate keys, unpaired surrogates, blank lines, CRLF, and non-canonical JSON
  are rejected;
- `previous_record_sha256` is `null` for the first record and the exact previous
  `record_sha256` thereafter;
- `record_sha256` is SHA-256 of the canonical record with only
  `record_sha256` omitted.

The chain detects changed, reordered, or removed middle records. Git review and
the last published `record_sha256` provide the external head pin needed to
detect removal of a suffix; a hash chain by itself cannot prove that its last
line was never truncated.

## Event draft schemas

`append` reads a strict JSON object containing every normal record field except
`previous_record_sha256` and `record_sha256`; the tool injects those two values.
The common draft shape is:

```json
{
  "schema_version": "1.0",
  "record_type": "architecture_iq_deployment_event",
  "event_type": "candidate_attested",
  "deployment_key": "quiz-staging-20260712-001",
  "recorded_at": "2026-07-12T12:00:00Z",
  "recorded_by": "github:maintainer",
  "reviewed_by": null,
  "facts": {}
}
```

Exact `facts` shapes are:

```text
candidate_attested
  release_id
  manifest {path, sha256, question_count}
  registry {path, sha256, registry_id, manifest_sha256,
            question_count, choice_count}
  source {repo_url, branch, commit, entrypoint}
  rollout {phase="report-app", fingerprint,
           preflight {path, sha256}}

deployment_declared
  environment, target_label, provider, project_id, deploy_id, site_url,
  backend_project_id, ingest_origin_sha256, report_origin_sha256

postgres_accepted / roundtrip_accepted / source_mapping_attested
  evidence {path, sha256}
  summary {event-specific safe fields copied exactly from validated evidence}

activated
  {}

superseded / rolled_back
  reason, replacement_deployment_key
```

Every evidence summary includes the same derived `deployment_context_id` plus
environment, target, app provider/project/deploy/site, backend project, and
ingest/report origin SHA-256 values. The context ID hashes those fields together
with deployment key, release, manifest, registry, and source commit. The CLI
requires event-specific safe summaries so reviewers can index and read the
ledger without opening evidence bodies. It independently derives or cross-checks
those summaries and rejects discrepancies. Context binding prevents accidental
cross-environment assembly; where the underlying tool omits target identity, the
binding is still explicitly reviewer-governed rather than provider-authenticated.

The provider mapping evidence envelope has this exact top-level schema:

```text
schema_version="1.0"
evidence_type="architecture_iq_provider_deployment_mapping"
captured_at
mapping_authority="reviewed_provider_control_plane_capture"
provider_export {path, sha256, media_type}
provider, project_id, deploy_id, site_url
environment, target_label, backend_project_id
ingest_origin_sha256, report_origin_sha256, deployment_context_id
deployed_at, deployment_status="ready"
repo_url, branch, source_commit, entrypoint
release_id, manifest_sha256, registry_id, rollout_input_fingerprint
```

The raw export may be strict JSON, UTF-8 text, PNG, or PDF and has its own
repository-relative path and raw hash. It must come from provider deployment
history/control-plane access and be manually reviewed; do not populate it from
values reported only by the application being audited. A hand-written mapping
envelope without the separately hash-bound capture is rejected. Even with that
capture, the ledger claims reviewed provenance, not a provider signature.

## CLI

Verify the complete chain and all referenced evidence:

```bash
PYTHONPATH=. .venv/bin/python tools/deployment_ledger.py verify \
  --repo . \
  --ledger deployments/ledger.jsonl
```

List safe per-deployment states:

```bash
PYTHONPATH=. .venv/bin/python tools/deployment_ledger.py list \
  --repo . \
  --ledger deployments/ledger.jsonl
```

Append one reviewed draft:

```bash
PYTHONPATH=. .venv/bin/python tools/deployment_ledger.py append \
  --repo . \
  --ledger deployments/ledger.jsonl \
  --event-json /path/to/reviewed-event-draft.json \
  --confirm-append
```

Without `--confirm-append`, no ledger, lock, directory, or placeholder record is
created. With confirmation, the command takes an exclusive lock, verifies the
entire existing chain, injects the previous/hash fields, verifies the proposed
new chain, writes one fsynced same-directory replacement, and prints the new
record hash. Commit the ledger and evidence only after `verify` passes and code
review confirms the provider evidence source.
