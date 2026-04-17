# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1078b1a70bcd22daa0ead376beedaa52bfe2cf8765ec6a491cb29b47f4429da`
- fixture hash: `sha256-48416b4518f830c212c5a38183605df066ce4a1235bd3582b824c27bcab21c53`
- score hash: `sha256-d5b59879a4d010be17918d1ee02fd78572fcf8beb4a373d4e96106f7843ad410`
- bundle hash: `sha256-599c81371cb47bb0e602dfc57dd0bf5df03f2087de1c4eb849216629b39ddd66`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57e6cc1ff0fcf88903029010179cd9e85affa629951b704a6bd53f2a38e4810e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-61b1f245b6c00d79de878107bf67d11ba2e8a05f029e0298660b53b6f96025c1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-796a0749ff2eda36a45bd5ebe96a16af12725628b7338b7fe7ae6cc82ac160af |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4aba8512be4a498600cb85956f514f9391c23e5f29c50655c614c312abbcfeb0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13b70918 | sha256-b5f92a80eea582020f355769d1f45f9e255cad5e615c55bb1c0a5bbea0db7216 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13b70918 | sha256-b5f92a80eea582020f355769d1f45f9e255cad5e615c55bb1c0a5bbea0db7216 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4071a301 | sha256-a657409b66906862b9fcbd379d9f1684de1fb233a3e3372a3410d11edfd0c38e |
