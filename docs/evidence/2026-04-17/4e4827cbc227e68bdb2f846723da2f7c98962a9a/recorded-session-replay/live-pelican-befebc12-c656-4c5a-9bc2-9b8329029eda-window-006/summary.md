# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f9684c38d91e55a983d42052df21e03bec407bc3f34393946fcda8e1b2d39f4`
- fixture hash: `sha256-5b6e1bbde60f4bcca2052f19249d943d07695521da1e7e8b46846e97b143bb5b`
- score hash: `sha256-d62af3c004a7d6e39cf7935baf7a15eb7d653918e44266901a553509fe49129c`
- bundle hash: `sha256-11c3e540b79fe29b11f459f7cb4dcd1f2513c7e7acf91267d94edb5dea23c4da`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a0ec494cc9b5ec66bff70ca0bc3e9262d5754f8a93ce7d222367da206ee232 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-874d74f2afeabefeb9ef909cbc1e1134bf8970995a0c90c388bf6f630195918d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7b81505a09b90327f1cad5c9fa830fab953d9ae96302941324402755a8002679 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-fc220cc4f9f6aca9131526a46db78c1e705700fee25370940b0bc465aa602d91 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d2fbf185 | sha256-8ace5e0efb58e5b893581454560a94f12a2760e6d3282d04378ee7a2eb1c5526 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d2fbf185 | sha256-6af9aacf6df4d3add110945468c95c27e3153367e7c3a96e395763f0fa48bfc2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-178de950 | sha256-efe1c0f141ce4a52f391ece564f98dd747176772b66c0dcfe1a3741478d0e591 |
