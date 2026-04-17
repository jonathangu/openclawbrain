# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30a54314d984e83263bc7ddfcb852ce4d67a835461588938c047eabba74d7daa`
- fixture hash: `sha256-a669f6ac0947e4907b9b5ff0ba78d765904f903d2ac7c540eba1f40434878bd9`
- score hash: `sha256-be7400c2bfb1059d25301dde31d25bb7962b3cef835be16e940241064d285c2c`
- bundle hash: `sha256-ffcf576c5dbb045366f8ba411eff9865a1435d052400c01cc0f0eb2d1e885f4c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ddef39ad20fc1c3136dfb625c29bf78d555d4df3233592558f3107ec01752a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0802a1b324b77813ed9f614c1e52eb2df034f6fa69a624781907bfb90ff5f0c2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b8a440b1872c2724b758f2b184265ef668545173c31b617f26f74e2ffeb0ab61 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2690d5aa29c9204d2fb148257bdc28a0ac7563cc70a5a7a5a409fdace64097af |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6dd203e3 | sha256-a5692a717c05b611c7fb08fb677c2eceff84298cc2f156168613f5bca795c291 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6dd203e3 | sha256-97cf313f22b7feb4b04d4d4d42c744f49f8f04d86ecfb0422799771d8277ece2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-76e7b452 | sha256-3da95c6c8d3a3bc13d60330ce507476502510b5f984727977bee87ffe89eb60a |
