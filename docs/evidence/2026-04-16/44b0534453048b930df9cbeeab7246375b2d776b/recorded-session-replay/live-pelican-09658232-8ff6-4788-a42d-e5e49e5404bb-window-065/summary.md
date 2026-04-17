# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b834ef975b4127c9fa6cce4b12fc80a10ac45f8451c226003f814696763d3404`
- fixture hash: `sha256-2ef4285ae644d199abac210f4e94c99bfd3cbffd40a56868154cea15ccdb9a86`
- score hash: `sha256-b1eaab09b3c0cf92b977da6762f7373b0ca469c7e65040e409e774e90a1c3816`
- bundle hash: `sha256-88243de3317130958cee75980a9bc096fc8c52df09db4c1bdad0bff5b8f1574c`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f63fdc694dc12e3fd1585f1a9d1a8d63286b83507ef0e36210c868d071e50d26 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c6876345f87396973862667c8ae9c1f649e854e46e15c115757eaba27196f144 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e4e20f2af4df7eabb2acdbb51602e1ac7c0823746925b1138baab93bd5557480 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5ec96ce98ecbdf5fd8fb244f132b087787b6925328e2fa0bdd992fda30e64afb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b422b063 | sha256-2e571aabe722cf198488942c28e35207648eb2ad9c3167f4dbb7698b6e11b493 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b422b063 | sha256-c0c7ba6d1df2255ec73c1aec50af2d5cacb0522206ec9f8576b191789cbfeee2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-123d092e | sha256-c5a329c5f10e983a3151682da1f92fab66e6c071abfe3e036571f79e8aa8187e |
