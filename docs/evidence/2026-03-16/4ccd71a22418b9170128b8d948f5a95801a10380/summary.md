# OpenClawBrain validation summary

- commit: `4ccd71a22418b9170128b8d948f5a95801a10380`
- validation mode: sterile-lane
- config path: `/Users/cormorantai/.openclaw-ocbphase1/openclaw.json`
- state dir: `/Users/cormorantai/.openclaw-ocbphase1`
- workspace: `/Users/cormorantai/.openclaw-ocbphase1/workspace-fixture`
- artifact dir: `/Users/cormorantai/openclawbrain/docs/evidence/2026-03-16/4ccd71a22418b9170128b8d948f5a95801a10380`

## Assertions
- teachRetrieval: {"taughtNodeId":"bn_2000f3c7-793","packVersion":17,"retrievedCorrectionVisible":true,"traceIncludesTaughtNode":true,"retrievedPackVersion":17}
- workerDownFailOpen: {"servedBeforeCrash":true,"servedPullRequestGuidanceBeforeCrash":true,"workerHealthyAfterCrash":false,"workerLastExit":{"code":null,"signal":"SIGKILL","at":1773675824182},"currentPackVersion":8,"servedAfterCrash":true,"servedPackVersion":8,"servedPullRequestGuidance":true}
- recurrentQuery: {"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"episodeId":null,"traceQueryText":null,"workerMode":"child","workerPid":null,"workerHealthy":false,"workerLastHeartbeatAt":null,"currentPackVersion":12,"aborted":false}
- shortLookup: {"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"lastAssemblyMode":"use_brain","aborted":false,"bypassEvidence":"Since `PLAYBOOK.md` does not exist in the provided project context files, would you like to create a new `PLAYBOOK.md` file? Here is an example of how you can create it using the `write` tool:\n\n```jso"}
- shadowMode: {"shadowMode":true,"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"episodeId":null,"traceQueryText":null,"injectedContextVisible":false,"aborted":false}
- noEmbedding: {"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"lastAssemblyMode":"use_brain","aborted":false}
- uninitialized: {"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"lastAssemblyMode":null,"aborted":false}

## Skipped
- worker-down: Host-surface worker-down assertion requires child-worker mode with a live worker PID.
- brain-teach: Phase-1 harness still needs a deterministic host-surface path for brain_teach assertion wiring; raw openclaw agent --local text prompting does not force tool use honestly.
