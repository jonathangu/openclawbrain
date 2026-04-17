# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-a72e56685dfa4084e9ab31e65868d232f8378cad212a5a1f0e641e4d8eab2d40`
- bundle hash: `sha256-f690431a04620adb66c320d5f1848832ba74b85440f51e3a6c48d07bddbf5f79`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c8e99c08fa9a9f740c3e6451db85cc24a1dd5f5dbcbad1273787fadddbbd230e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-82d432ffe14342a2b4fe2ce75f991726119384cf9bad639548d9e11008cc6a47 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-320834c6d4ba4dd5bb61ef01aba17b0af1a5a004e48964c1ec853de401ebceac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e87fd86c | sha256-36ff2411ecff9775eacca58450f369e361ef289e2db090d99426bcf10aeadc31 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e87fd86c | sha256-9ee28f43617df90c2e736670a64c0737d6a2b6cb1f1ac086faaca700b0961ace |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-21f49e07 | sha256-7836a2b92a501870a519eafb321c52c14ed8e1989a1a0747168ec6bc693f8674 |
