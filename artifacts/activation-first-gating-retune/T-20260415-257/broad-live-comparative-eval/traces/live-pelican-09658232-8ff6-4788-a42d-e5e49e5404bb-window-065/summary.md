# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b834ef975b4127c9fa6cce4b12fc80a10ac45f8451c226003f814696763d3404`
- fixture hash: `sha256-2ef4285ae644d199abac210f4e94c99bfd3cbffd40a56868154cea15ccdb9a86`
- score hash: `sha256-dc7eb4efe9220ea6bf6f30e3c6d7ae559e26380d6a4b9c85ed894a0606c3eea9`
- bundle hash: `sha256-4a8179f5cadd122d9b61e8340883aab5cc75b3257df479225e40489a82f59fd5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f63fdc694dc12e3fd1585f1a9d1a8d63286b83507ef0e36210c868d071e50d26 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-538831c5247fba040f3bd03663520b1ce9984cef173eb869872d1482d6a142a0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd82fe3170f3ea16f6c06202f75793399befef2334c700117e2948c2efc79fe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f44ce8e41c16ccf753304a84eacca5f2f183e1327bea9d64e1764fcc5863a0ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7de46e21 | sha256-fb54f60ce67e625a2375a7f0fbe1f93999216e28dea204e5559c65a8440697c5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7de46e21 | sha256-343954e91b8af5d646e92b07603fbe7e078aa76b97633af2b66b8e43b1069808 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-dbfec6ec | sha256-00300028a135c69fcf888a12d2f80cc5caf996730c6a781b5e232744701a7fab |
