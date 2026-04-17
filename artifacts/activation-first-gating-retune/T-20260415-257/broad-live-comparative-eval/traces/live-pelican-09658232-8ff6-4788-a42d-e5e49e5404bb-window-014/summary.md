# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a2e597e92fe22d4f55c094b8ed54b6a9af6fa4591283d89702e798da892600a7`
- fixture hash: `sha256-622fac4fb2f464038d17b973948a3daa701456585a35960e995213dcda72d3b1`
- score hash: `sha256-6a14bdc1aff8ec87a52bff5b818dc6040019fbb88467945762af1a81b72e65f6`
- bundle hash: `sha256-d949bd35ae003295f6e8422c7e01032c545266021b908093ed0e687226f8cda1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-937c7f8b3de6cb0ba567e2def00dcbf253af96c301f9c26d07a7c1aa6375230e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d61a22c331fc8e5f84543df0f47847e9e291d87c1cef0f7bfc1d52693fec5ce5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8b8848419dd145dbffcfe04695fa598e7ccf5ed0614946595cd46c5e3822d0ff |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f936d3f6381620dbb8db419d12ff18948572621f184caf6e826d931b873726da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a11770c4 | sha256-6db31ec7dc1ccb1898d1482adeaf0c400ff0ce5c5ba9ef2fd89fc8b82e7d5a0e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a11770c4 | sha256-48fe8e32a51472b76ab858be5a29d2592dee032751f4b1552c0d6b6d16f1b563 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6533528d | sha256-77835e0834eaafab0391c60c4cf25618e34a7ff95931d507d203adac72167843 |
