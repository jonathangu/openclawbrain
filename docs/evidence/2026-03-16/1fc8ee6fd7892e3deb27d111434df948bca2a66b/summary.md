# OpenClawBrain validation summary

- commit: `1fc8ee6fd7892e3deb27d111434df948bca2a66b`
- validation mode: sterile-lane
- config path: `/Users/cormorantai/.openclaw-ocbphase1/openclaw.json`
- state dir: `/Users/cormorantai/.openclaw-ocbphase1`
- workspace: `/Users/cormorantai/.openclaw-ocbphase1/workspace-fixture`
- artifact dir: `/Users/cormorantai/openclawbrain/docs/evidence/2026-03-16/1fc8ee6fd7892e3deb27d111434df948bca2a66b`

## Assertions
- teachRetrieval: {"taughtNodeId":"bn_8c0d6437-cc6","packVersion":7,"retrievedCorrectionVisible":true,"traceIncludesTaughtNode":true,"retrievedPackVersion":7}
- workerDownFailOpen: {"servedBeforeCrash":true,"servedPullRequestGuidanceBeforeCrash":true,"workerHealthyAfterCrash":false,"workerLastExit":{"code":null,"signal":"SIGKILL","at":1773674656519},"currentPackVersion":3,"servedAfterCrash":true,"servedPackVersion":3,"servedPullRequestGuidance":true}
- recurrentQuery: {"validationRecordCountBefore":2,"validationRecordCountAfter":2,"mode":null,"traceId":null,"episodeId":null,"traceQueryText":null,"workerMode":"child","workerPid":null,"workerHealthy":false,"workerLastHeartbeatAt":null,"currentPackVersion":6,"aborted":false}

## Skipped

## Failure
- Validation harness expected recurrent host-agent query to emit a host-surface validation record, but none was captured.
