# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-79532f3b0ed4010e846f65618be48c2307af13b97dd581f294dd9d5e6325f8eb`
- fixture hash: `sha256-6cb2c0584a4478c43146057da013c5b788958c7763f4cd2e66653360656b5ed8`
- score hash: `sha256-755dcefe41b569781b081503a9d261fa3c58ed3b646b54407b632646a2a8961e`
- bundle hash: `sha256-b0fa69d4dcf2f53937ec37222d3d69f673516af5d1452d4ea47cff38b3ad654c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9133ee007dc762137868e0af1d2b1845ed239e3a386c6ccc9187f6b355e2ae22 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-750f5e60ddb5843ec51b6ddcc8db5420db84e658d52a063d05821db7fe181032 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b3c5ac821fc8a378a75c0284ee44399ddfb90b910114a4bb60a1213c16557033 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cc4e657445598556c3b4e99a8745274311c68ca3bbb82e1b6025350c07a0ac0d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-110393bc | sha256-987c48c50af794d53bff929df86fa7cf4275f022930eac3e355b907a26d7e213 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-110393bc | sha256-44c779cc2b62d3fb1f7f4b07e551f6ad1aa36a88bdbfadf8dbdc6f7bb6d74bed |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-110393bc | sha256-987c48c50af794d53bff929df86fa7cf4275f022930eac3e355b907a26d7e213 |
