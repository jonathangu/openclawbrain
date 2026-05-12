# OpenClawBrain Codex Continuity First-Try Fix Plan

Status: fix plan after first real Telegram-to-Codex operator trial  
Date: 2026-05-12  
Owner: OpenClawBrain  
Related release: `openclawbrain@0.2.32`

## Executive Summary

The first end-to-end Codex continuity trial proved the architecture is worth continuing.

OpenClawBrain successfully:

- read a Codex Desktop thread from local Codex state;
- bound a Telegram chat to a specific Codex thread;
- showed recent assistant messages;
- tailed thread output into Telegram;
- injected a mobile message into the live Codex thread;
- steered the active Codex turn;
- unbound and unwatched the thread afterward.

That is the hard proof. Telegram can become a real low-bandwidth operator console for Codex UI without modifying OpenClaw core.

But the trial also showed that the current UX is too sharp for daily use. The bridge currently exposes powerful primitives, but the product shape is not yet forgiving enough. The biggest issue is semantic: `/brain codex reply` feels like "paste this note into the thread," but in Codex it actually means "start or continue a real Codex turn." That can edit files, run tools, request approvals, and leave dirty work in a repo.

The next fix pass should make the bridge operator-grade:

> Observe should be effortless. Notes should be passive. Acting should be explicit. Steering should be reserved for live work. Detach should be one obvious escape hatch.

## First-Try Verdict

Prototype grade: `6.5/10`  
Polished operator UX grade: `4/10`  
Architecture proof grade: `8/10`

The shipped primitives are real. The workflow is promising. The fix is not to retreat from Telegram-to-Codex control. The fix is to make intent explicit, reduce noise, harden readiness checks, normalize identity, and add a clean detach path.

## What Worked

- `/brain codex status` worked.
- `/brain codex binding` worked.
- `/brain codex last` worked.
- `/brain codex tail` worked.
- `/brain codex watch` worked.
- `/brain codex reply` and `/brain codex send` worked once the correct Codex app-server was listening.
- `/brain codex steer` worked against the same active turn.
- `/brain codex unbind` worked.
- `/brain codex unwatch` worked.
- Binding Telegram to thread `019e1294-4aa2-7280-8c15-2315c81256e3` worked.
- The read path worked through Codex SQLite and rollout JSONL even when the app-server was not running.
- The write path worked after the Codex Desktop app-server was available at `ws://127.0.0.1:53177`.
- The mobile message was accepted into Codex as turn `019e1e3d-2be2-7780-b431-80a8c5f81a87`.
- Cleanup was possible after the fact by unbinding and pausing watches.

## What Went Badly

### 1. App-Server Readiness Was Brittle

After a gateway restart, nothing was listening on `53177`. The read path still worked because SQLite fallback worked, but writes failed until the Codex Desktop app-server was manually started.

This made the product feel broken even though the bridge logic was mostly correct.

The user-facing issue:

```text
Reads work. Writes fail. The reason is hidden.
```

The correct behavior:

```text
Reads work through local state.
Writes are not ready because Codex app-server is not reachable at ws://127.0.0.1:53177.
Start command / LaunchAgent / last error shown plainly.
```

### 2. Config Update Was Clunky

The protected config patch path refused the intended update, so the setup fell back to direct config editing.

This is acceptable for emergency local setup, but not acceptable as the normal install/update path. The bridge needs a clean "doctor/fix" flow that can:

- inspect config;
- explain missing fields;
- apply safe profile-local changes;
- avoid dirtying OpenClaw core;
- restart the gateway if needed;
- verify runtime load afterward.

### 3. Watch Mode Was Too Noisy

Assistant messages kept forwarding into Telegram once the watched Codex thread started doing real work.

That proved message watches work, but it also showed the default is too loud. Telegram should not become a second coding UI. It should provide controlled operator awareness.

The correct default:

- completion;
- blocker;
- failure;
- approval-needed;
- auth failure;
- explicit `tail` for message-level forwarding.

Assistant-message streaming should be opt-in, time-limited, and easy to stop.

### 4. Binding Identity Was Not Canonical

Both of these shapes appeared:

```text
telegram:8518484672
8518484672
```

That creates duplicate bindings, duplicate watch behavior, confusing cleanup, and possible wrong-target behavior.

The bridge needs one canonical Telegram chat identity representation and a migration for older rows.

### 5. `/reply` Behaved Like An Action, Not A Passive Note

The injected mobile text was interpreted by Codex as a real task request. Codex began editing `/Users/cormorantai/cormorantai-mvp`, modifying:

- `src/app/page.tsx`
- `tests/bihua-alert-filter.test.ts`

Then the target Codex hit an approval/test issue:

```text
SecItemCopyMatching failed -50
```

This is exactly the wrong flow if the operator only meant:

```text
Attach this observation/context to the thread.
```

Today the bridge has observe, act, steer, and detach primitives. It is missing a clear passive note primitive.

### 6. Cleanup Required Too Much Knowledge

The operator needed to know both:

- `/brain codex unbind`
- `/brain codex unwatch <thread-id>`

That is not humane enough for a mobile control surface. There should be one obvious escape hatch:

```text
/brain codex detach
```

Detach should stop all Telegram coupling for the current chat unless the user explicitly asks for narrower cleanup.

## Product Model

The bridge should expose five intent classes.

### Observe

Read state without changing Codex.

Examples:

```text
/brain codex status
/brain codex threads
/brain codex binding
/brain codex last --bound
/brain codex messages --bound --limit 5
/brain codex handoff --bound
```

Guarantee:

> Observe commands never start a Codex turn and never write to Codex.

### Follow

Forward selected future Codex events into Telegram.

Examples:

```text
/brain codex tail --bound
/brain codex watch --bound
```

Guarantee:

> Follow commands create temporary watches. Default watches are quiet; message-level tailing is explicit and time-limited.

### Attach Note

Attach passive context to the bridge's understanding of a Codex thread without asking Codex to act immediately.

Proposed command:

```text
/brain codex note <message>
```

Guarantee:

> Notes do not call `turn/start`, do not call `turn/steer`, do not edit files, do not run tools, and do not appear as instructions to Codex unless the user later explicitly includes them in an action.

### Act

Send a real instruction into Codex that may create a new turn, edit files, run tools, or request approvals.

Examples:

```text
/brain codex act <message>
/brain codex reply <message>
/brain codex send <thread-id|--bound> <message>
```

Guarantee:

> Act commands are explicit about side effects and create an audited outbound event.

### Steer

Redirect an active in-progress Codex turn.

Example:

```text
/brain codex steer --bound <message>
```

Guarantee:

> Steer requires an active turn. If no active turn exists, the bridge tells the operator to use `act` or `reply` instead.

### Detach

Stop Telegram coupling quickly.

Example:

```text
/brain codex detach
```

Guarantee:

> Detach unbinds the current Telegram chat and pauses all active watches for that chat.

## Desired Command Surface

The common mobile UX should be small.

```text
/brain codex status
/brain codex threads
/brain codex bind <thread-id>
/brain codex last
/brain codex tail
/brain codex note <message>
/brain codex act <message>
/brain codex steer <message>
/brain codex detach
```

Power-user commands can remain available, but they should not dominate documentation.

```text
/brain codex messages [thread-id|--bound|--latest] [--limit N] [--role assistant|user|all]
/brain codex watch [thread-id|--bound|--latest] [--messages] [--ttl 15m]
/brain codex watches
/brain codex unwatch <watch-id|thread-id|latest|all>
/brain codex send <thread-id|--bound> <message>
/brain codex unbind
/brain codex handoff [thread-id|--bound|--latest]
```

## Fix 1: Add `/brain codex detach`

### Problem

The operator needs one obvious way to stop the bridge from forwarding messages or targeting a thread.

### Behavior

```text
/brain codex detach
```

For the current Telegram chat:

1. Remove the active conversation binding.
2. Pause active watches whose `notify_target` matches the chat.
3. Pause both thread-specific watches and `latest` watches for that chat.
4. Leave historical audit records intact.
5. Return one concise cleanup report.

Example response:

```text
Detached Codex from this Telegram chat.

Stopped:
- binding to thread 019e1294-4aa2-7280-8c15-2315c81256e3
- 2 active watches

Nothing else will be forwarded here unless you bind or tail again.
```

### Related Commands

```text
/brain codex detach --all
/brain codex detach --thread <thread-id>
/brain codex detach --keep-binding
/brain codex detach --keep-watches
```

Only plain `detach` should be needed most of the time.

### Implementation Notes

Add a `detachCodexChat` store method that runs in one transaction:

- find canonical chat key;
- delete or mark binding inactive;
- pause matching active watches;
- write `codex_bridge_events` rows:
  - `binding_detached`
  - `watch_paused_by_detach`
  - `detach_completed`

### Tests

- Detach removes binding for current chat.
- Detach pauses all active watches for current chat.
- Detach does not affect another chat's watches.
- Detach is idempotent.
- Detach reports zero changes cleanly.
- Detach handles canonical and legacy chat keys.

## Fix 2: Normalize Telegram Chat Identity

### Problem

The bridge currently allowed duplicate chat-key shapes:

```text
telegram:8518484672
8518484672
```

That creates duplicate bindings and confusing cleanup behavior.

### Canonical Rule

Every Telegram chat key should be stored as:

```text
telegram:<chat_id>
```

Every Telegram sender key should be stored as:

```text
telegram-user:<sender_id>
```

If `message_thread_id` exists, the scoped chat key should be:

```text
telegram:<chat_id>:topic:<message_thread_id>
```

### Compatibility Rule

On read, the bridge may resolve legacy keys.

On write, the bridge must store only canonical keys.

### Migration

Add a lightweight bridge DB migration:

1. Scan `codex_conversation_bindings`.
2. Convert bare numeric Telegram chat ids to `telegram:<id>`.
3. Merge duplicates by keeping the newest active binding.
4. Mark older duplicate bindings as superseded or delete if the table only stores one active row.
5. Scan `codex_bridge_watches`.
6. Convert `notify_target` bare ids to canonical ids.
7. Deduplicate active watches with the same chat/thread/scope/classes.
8. Record redacted migration events.

### Tests

- Bare numeric chat id resolves to canonical binding.
- New bind stores canonical chat id.
- Duplicate legacy/canonical bindings are merged.
- Detach cleans both legacy and canonical rows.
- Watches dedupe across legacy/canonical targets.

## Fix 3: App-Server Readiness Diagnostics

### Problem

The bridge can read through SQLite while writes require Codex app-server. The current status output does not make readiness obvious enough.

### New Status Output

`/brain codex status` should include:

```text
Codex continuity:
- reads: ready via SQLite fallback
- app-server: reachable at ws://127.0.0.1:53177
- writes: ready
- steering: ready when active turn exists
- bound thread: 019e1294-...
- active watches: 1 quiet, 0 tail
```

If app-server is down:

```text
Codex continuity:
- reads: ready via SQLite fallback
- app-server: not reachable at ws://127.0.0.1:53177
- writes: not ready
- steering: not ready

Fix:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclawbrain.codex-appserver.plist
```

### Add `/brain codex doctor`

`doctor` should check:

- plugin version;
- gateway plugin load status if available;
- configured `appServerUrl`;
- whether the WebSocket accepts a connection;
- whether `initialize` succeeds;
- whether `thread/list` or `thread/read` works;
- whether SQLite fallback is readable;
- whether `threads.rollout_path` exists for recent threads;
- whether writes are enabled;
- whether steer is enabled;
- whether trusted Telegram sender/chat is configured;
- whether repo allowlist covers the bound thread.

### Add `/brain codex fix-readiness`

For local Jonathan profiles, this command can be a guided operator command or CLI script, not necessarily a Telegram command.

It should:

- install or refresh the LaunchAgent plist;
- set `codexBridge.appServerUrl`;
- set `codexBridge.preferAppServer`;
- restart the app-server LaunchAgent;
- restart the OpenClaw gateway;
- re-run doctor.

### Tests

- Status says writes not ready when app-server URL is missing.
- Status says writes not ready when WebSocket connection fails.
- Status says writes ready when app-server handshake succeeds.
- Doctor distinguishes read readiness from write readiness.
- Doctor includes last app-server error without leaking raw transcript content.

## Fix 4: Split Passive Note From Action

### Problem

`/brain codex reply <message>` starts a real Codex turn. The operator may intend only to attach context.

### Proposed Command

```text
/brain codex note <message>
```

### Behavior

The note is stored in the bridge DB as thread-attached operator context:

- thread id;
- canonical chat key;
- sender key;
- timestamp;
- redacted preview;
- full local text only if local bridge config allows note body storage;
- status: `active`, `included_in_action`, `dismissed`, or `expired`.

Notes should appear in:

```text
/brain codex handoff --bound
/brain codex notes --bound
/brain codex status
```

Notes should not be injected into Codex automatically.

### Optional Later Behavior

If Codex app-server eventually supports passive annotations, OpenClawBrain may mirror notes into Codex as non-actionable thread annotations. Until then, notes remain OpenClawBrain-owned.

### Acting With Notes

Add:

```text
/brain codex act --with-notes <message>
```

This sends the user's action plus active notes into Codex in a clear block:

```text
Operator note context:
- ...

Instruction:
...
```

After accepted send, notes become `included_in_action`.

### Warning Copy

For `reply`, `send`, and `act`, use product language that makes side effects explicit.

First-use or ambiguous response:

```text
This will start a real Codex turn and may edit files, run tools, or request approvals.

Use:
- /brain codex note <message> for passive context
- /brain codex act <message> to proceed
```

For trusted local happy path, `/brain codex act` should send directly. The command name itself carries the intent.

### Command Compatibility

Keep `/reply` for power users, but change public docs to prefer:

- `note` for passive context;
- `act` for a real new turn;
- `steer` for live interruption.

### Tests

- `note` does not call app-server.
- `note` is visible in handoff.
- `note` expires or can be dismissed.
- `act --with-notes` includes active notes once.
- `reply` remains available but warns or maps to action semantics according to config.
- Notes do not become durable memory by default.

## Fix 5: Quieter Watch Defaults

### Problem

Message-level watches can spam Telegram when Codex is actively working.

### New Defaults

Plain watch:

```text
/brain codex watch --bound
```

Should notify only:

- completion;
- blocker;
- failure;
- approval required;
- auth failure.

Message tail:

```text
/brain codex tail --bound
```

Should forward completed assistant messages, but with:

- a TTL;
- chunking cap;
- a visible stop command;
- dedupe;
- no tool chatter by default.

### Default TTLs

```text
watch terminal events: 24h
tail assistant messages: 30m
tail --long: 2h
tail --until-complete: until terminal event, capped at 6h
```

### Tail Start Response

```text
Tailing completed assistant replies for this Codex thread for 30 minutes.

I will keep tool chatter quiet.
Stop with /brain codex detach or /brain codex unwatch <watch-id>.
```

### Tests

- Plain watch does not forward assistant messages.
- Tail forwards assistant messages.
- Tail expires automatically.
- Tail dedupes repeated rollout records.
- Tail chunks large messages.
- Tail does not forward tool deltas.
- Detach stops active tail watches.

## Fix 6: Write Warning And Risk Copy

### Problem

Telegram-originated writes can trigger edits, tests, approvals, or dirty worktrees. The bridge needs to set expectations without blocking the local happy path.

### Policy

The bridge should classify every write:

- `note`: no Codex side effect;
- `low`: explanation/status instruction;
- `medium`: coding action in allowlisted repo;
- `high`: publish, deploy, delete, secrets, auth, prod, full-access, destructive operations.

### UX

`note` never warns about Codex side effects because it has none.

`act` in trusted local profiles can send directly for low/medium risk, but the response should still say what happened:

```text
Sent action to Codex thread 019e1294...
Codex may edit files or run tools under its normal approval/sandbox rules.
```

High risk remains refused from Telegram unless explicitly configured:

```text
I will not publish, deploy, delete, or handle secrets from Telegram in this profile.
Use Codex UI at the computer.
```

### Tests

- Low/medium trusted bound action sends.
- High-risk action refuses by default.
- Risk copy appears in accepted action response.
- Untrusted sender refuses before risk classification details leak.

## Fix 7: Consequence Cleanup Workflow

### Problem

The first trial caused real target repo changes in `/Users/cormorantai/cormorantai-mvp`.

OpenClawBrain should not auto-revert unrelated repo changes, but it should make consequences visible.

### Add Post-Write Awareness

After a Telegram-originated write, the bridge should be able to say:

```text
Codex accepted the turn.
Possible side effects are now Codex-owned.
Use /brain codex handoff --bound to inspect observed repo/thread state.
```

If repo status can be cheaply inspected through allowed local context:

```text
Observed repo now has uncommitted changes:
- src/app/page.tsx
- tests/bihua-alert-filter.test.ts
```

This should be an observed fact, not a memory.

### Add Handoff Section

`/brain codex handoff` should separate:

- observed facts;
- Codex-reported claims;
- operator notes;
- possible side effects;
- next choices.

Example:

```text
Observed repo state:
- 2 uncommitted files

Codex-reported claim:
- test hit SecItemCopyMatching failed -50

Operator choices:
- continue in Codex UI
- send /brain codex steer --bound "stop and summarize only"
- send /brain codex detach
```

## Fix 8: App-Server Launch And Recovery

### Problem

The bridge depends on a local Codex app-server, but the dependency was invisible.

### Local Setup Standard

Jonathan's local profiles should use:

```text
codexBridge.appServerUrl = "ws://127.0.0.1:53177"
codexBridge.preferAppServer = true
codexBridge.enableTelegramWrites = true
codexBridge.enableTelegramSteer = true
```

The LaunchAgent should run:

```text
/opt/homebrew/bin/codex app-server --listen ws://127.0.0.1:53177
```

### Recovery Behavior

If app-server is down:

- reads continue through SQLite fallback;
- writes report `not ready`;
- status shows exact readiness error;
- doctor shows exact recovery command;
- no write command should silently fall back to SQLite or JSONL.

### Tests

- Writes refuse when app-server is down.
- Reads still work when app-server is down.
- Status shows mixed readiness.
- Doctor gives the configured URL.
- No write path ever writes SQLite or rollout JSONL.

## Fix 9: Documentation Cleanup

### Problem

The current docs list too many commands. The useful product story is simpler.

### Public Language

OpenClawBrain should describe Codex continuity this way:

> Codex UI is the high-bandwidth workbench at the computer. OpenClaw on Telegram is the mobile operator surface. OpenClawBrain bridges them: ask what Codex is doing, read the last message, follow a thread briefly, attach passive notes, send explicit actions, steer live work, and detach when you are done.

### Website Examples

Use only a few clear examples:

#### Check What Codex Is Doing

```text
/brain codex status
```

#### Read The Last Reply

```text
/brain codex last --bound
```

#### Follow Briefly

```text
/brain codex tail --bound
```

#### Attach A Passive Note

```text
/brain codex note The failing test seems related to the auth mock, not the UI.
```

#### Ask Codex To Act

```text
/brain codex act Please fix the auth mock and rerun only the focused test.
```

#### Steer Live Work

```text
/brain codex steer --bound Stop editing and summarize what changed.
```

#### Detach

```text
/brain codex detach
```

### Docs To Update

- `docs/CODEX_TELEGRAM_FULL_BRIDGE_PLAN.md`
- `docs/CODEX_CONTINUITY_BRIDGE_OPERATOR_GUIDE.md`
- `README.md`
- `packages/openclaw-plugin/README.md`
- `openclawbrain.ai/codex-continuity/`
- `openclawbrain.ai/install/`

## Proposed Data Model Additions

### `codex_operator_notes`

```sql
CREATE TABLE codex_operator_notes (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  source_channel TEXT NOT NULL,
  source_chat_key TEXT NOT NULL,
  source_sender_key TEXT,
  body TEXT,
  redacted_preview TEXT NOT NULL,
  status TEXT NOT NULL,
  expires_at TEXT,
  included_outbound_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

Rules:

- `body` is local bridge DB only, never durable memory.
- public-safe audit should use `redacted_preview`.
- default expiry should be short, such as 24 hours.

### `codex_chat_identity_migrations`

```sql
CREATE TABLE codex_chat_identity_migrations (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  old_key TEXT NOT NULL,
  new_key TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
```

### `codex_detach_events`

This can be represented through `codex_bridge_events`; a separate table is not required unless analytics become useful.

Event types:

- `detach_requested`
- `binding_detached`
- `watch_paused_by_detach`
- `detach_completed`

## Implementation Phases

### Phase 1: Operator Cleanup

Ship:

- canonical Telegram chat identity;
- migration/dedupe for existing bindings and watches;
- `/brain codex detach`;
- quieter watch defaults;
- tail TTL;
- clearer status watch/binding counts.

Acceptance:

- one command stops forwarding;
- duplicate chat keys are gone;
- plain watch no longer forwards every assistant message;
- tail still works when explicitly requested.

### Phase 2: Readiness And Doctor

Ship:

- app-server readiness in `/brain codex status`;
- `/brain codex doctor`;
- local LaunchAgent verification;
- config verification across OpenClaw profiles;
- better error copy for write failures.

Acceptance:

- status distinguishes read-ready from write-ready;
- app-server-down error is obvious;
- no one has to remember why `last` works but `reply` fails.

### Phase 3: Passive Notes

Ship:

- `/brain codex note`;
- `/brain codex notes`;
- notes in handoff;
- `act --with-notes`;
- note expiry/dismissal;
- note storage boundary tests.

Acceptance:

- operator can attach context without starting a Codex turn;
- notes never call `turn/start`;
- handoff clearly labels notes as operator context.

### Phase 4: Action Semantics

Ship:

- `/brain codex act`;
- keep `/reply` as power-user alias or compatibility command;
- accepted-write side-effect copy;
- high-risk refusal copy;
- post-write handoff hints.

Acceptance:

- command names match intent;
- low/medium local trusted action path stays fast;
- high-risk Telegram writes remain controlled.

### Phase 5: Website And Operator Guide Refresh

Ship:

- simplified product-first site language;
- fewer commands on the public page;
- clear examples for observe, note, act, steer, detach;
- local operator guide with full command reference.

Acceptance:

- public page no longer feels like an API dump;
- daily commands are obvious;
- sharp commands are documented but not over-promoted.

## Test Plan

### Identity

- canonical chat key creation;
- legacy bare chat id lookup;
- legacy/canonical duplicate merge;
- topic-specific chat key;
- sender key normalization.

### Detach

- detach unbinds current chat;
- detach pauses current chat watches;
- detach does not affect other chats;
- detach handles no binding;
- detach handles no watches;
- detach is idempotent.

### Watch Quietness

- default watch only terminal/blocker classes;
- tail forwards assistant messages;
- tail excludes tool chatter;
- tail TTL expires;
- detach stops tail;
- duplicate records do not resend.

### App-Server Readiness

- status when app-server reachable;
- status when app-server down;
- status when URL missing;
- status when SQLite readable but app-server down;
- doctor shows exact configured URL and failure reason.

### Notes

- note stores without app-server call;
- note appears in handoff;
- note does not become durable memory;
- note redaction works;
- note expiry works;
- `act --with-notes` includes and marks notes.

### Actions

- `act` sends through app-server;
- `reply` compatibility path works;
- side-effect copy appears;
- high-risk text refuses by default;
- no `--latest` writes;
- exact/bound target required.

### Steering

- steer works with active turn id;
- steer refuses idle thread with helpful copy;
- steer remains separate from act.

### Regression

- `/brain codex status`;
- `/brain codex threads`;
- `/brain codex last`;
- `/brain codex messages`;
- `/brain codex bind`;
- `/brain codex binding`;
- `/brain codex unbind`;
- `/brain codex watches`;
- `/brain codex unwatch`;
- `/brain codex handoff`.

## Release Checklist

Before publishing the fix release:

- run full test suite;
- install into all local OpenClaw profiles;
- verify `/Users/guclaw/openclaw` remains clean;
- verify app-server LaunchAgent is running;
- verify `/brain codex status` shows write readiness;
- test `last`, `tail`, `note`, `act`, `steer`, and `detach` locally;
- package plugin;
- publish GitHub commit/tag/release;
- publish ClawHub package;
- install smoke from `clawhub:openclawbrain`;
- update openclawbrain.ai;
- update jonathangu.com if product positioning changes;
- verify live pages.

## Success Criteria

The next real trial should feel like this:

1. Jonathan opens Telegram.
2. He asks `/brain codex status`.
3. OpenClawBrain says what Codex is doing and whether writes are ready.
4. He binds a thread or uses the existing binding.
5. He reads the latest Codex reply.
6. If he wants passive context, he uses `note`.
7. If he wants Codex to work, he uses `act`.
8. If Codex is already working and needs redirection, he uses `steer`.
9. If Telegram gets noisy or the session is done, he uses `detach`.

The key product improvement:

> The operator should never accidentally start a real Codex task when they meant to attach a note.

The key reliability improvement:

> The operator should always know whether reads, writes, and steering are actually ready.

The key cleanup improvement:

> One command should stop the bridge from talking into that Telegram chat.

## Bottom Line

The first try proved OpenClawBrain can bridge Telegram and Codex UI. The next pass should not add a pile of new power-user commands. It should make the existing power safer and more obvious.

The product should move from:

```text
Here are raw primitives for reading, watching, writing, steering, unbinding, and unwatching.
```

to:

```text
Observe. Note. Act. Steer. Detach.
```

That is the shape of a daily-use operator console.
