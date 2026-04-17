# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-6f0f5a4e2bc60ac790dc0a7b69b1a478d5454d0ae049724095e61624711b6cc9`
- bundle hash: `sha256-920b28e6a504c099ec7e2baa5d015a59a64e235e6babac2c132552c50fb07e3e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-23f9ae274e15ddbc92ae3cb8e59779be164abe330585f210efc65c3c37ad0e80 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-bafe21258dd9806fed762b2b270d2efd70bb4d41e0a1de4d389217c34a59461f |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-3d4ec989b9f402e60c5e648730d205c2eec473492af480b1fbdecc94fef70aa3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-e385c05c | sha256-dde088deffbf52773554973fe59d8d64d1f3eb1afc0d3470e57b210a7d37c92e |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-e385c05c | sha256-f23d2de9f690e43c5bbe7cb633fa9fdaec68ad05ca41f1e6c29e0ba6314a9318 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-4cef291f | sha256-093177cbf984b6c02656506b1bc1389cb73d64f25cf8197ce2074d501e3dccae |
