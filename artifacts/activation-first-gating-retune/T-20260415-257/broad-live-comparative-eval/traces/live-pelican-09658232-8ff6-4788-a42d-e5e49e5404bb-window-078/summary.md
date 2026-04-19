# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8fecd38f3aa3470c67016a58c02da538613366240f311d73e765e2e999bfc5e1`
- fixture hash: `sha256-9a635fc4466dcd1f01d2e94228a353c7c6a97d36b77eaea2bf2676d0c4e0cb26`
- score hash: `sha256-97bb932ccaa51c9f858fd34e30213ba2caa7c718520ab1a1d8860ada2b709ee2`
- bundle hash: `sha256-51c01c39b91dd4822e07d618bf749b40ad7d2877869b30c10b46ba328ec41aaa`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c32d500730de7d73f2a2bf38e8b78d2d6ad04a3a58dd8029622c951f7ddee70 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-a7627bbfdab727032d0394386776f61800c982f30035cc1d93821ce442a00b38 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-b8c0aad87f56e0fa85e8fb8440b5358b2d0b445060d08a344b289c54c83df588 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-73803e80b50055c41cbd8d1de5bb96446fb64072009c17155e3c2add4b897445 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-68f361e8 | sha256-41f500ac0af65ff25f2ce53a07e34a4b2ec398bf083bc4e25a0a488bc54660ce |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-68f361e8 | sha256-ff6304c7d2ccc8867d2d035a31e0950cfc733edbc5eb1662cec3e21ff83c79c1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-68f361e8 | sha256-41f500ac0af65ff25f2ce53a07e34a4b2ec398bf083bc4e25a0a488bc54660ce |
