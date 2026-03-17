# OpenClawBrain 0.3.3

Published package: `@jonathangu/openclawbrain@0.3.3`

## What shipped

This patch fixes a real operator-facing regression in the supervised child-worker path.

On launchd-served installs like Eagle, the worker supervisor previously launched the child with:
- a bare `--import tsx/esm`
- no explicit working directory

When the parent gateway process was started with `cwd=/`, Node tried to resolve `tsx` from `/` and the child worker crash-looped with:

```text
ERR_MODULE_NOT_FOUND: Cannot find package 'tsx' imported from /
```

## The fix

`WorkerSupervisor` now:
- resolves the `tsx` loader to an absolute `file://...` path from the plugin's own module context
- launches the child with `cwd` pinned to the plugin root

That makes child-worker startup deterministic instead of depending on the service manager's cwd.

## Why it matters

`brainWorkerMode=child` is the truthful production boundary for OpenClawBrain.

Before this fix, an operator could install the plugin correctly, restart correctly, and still get a broken learner because the child never booted under launchd.
That meant the local workaround was `brainWorkerMode=in_process`, which kept the brain alive but was explicitly the wrong production mode.

With 0.3.3:
- Eagle is back on real `child` mode
- the worker starts successfully
- `workerHealthy=true` comes back from live status on the real Eagle profile

## Validation

This release was validated with:
- a focused test that reproduces the launchd-style `cwd=/` seam and proves the child boots without `tsx` resolution failures
- targeted runtime tests for the supervised child-worker boundary
- real Eagle profile validation after restart
- full `npm run release:verify`
