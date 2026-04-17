# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b224dc602b7429463a9b2fd5346afa6d3382bb3fd84bc9d3cceb0d3ff24896dc`
- fixture hash: `sha256-493fd471e0bb608979cd024ca51b9104b86ec7063e95845a4d6e7076002d21f4`
- score hash: `sha256-d755e284b013d2e39e238b6883607df003674d4601a727e06af955b8b8168750`
- bundle hash: `sha256-7f8d0dfa9f2b7deee94c101ff313229da443e452b24a33dcbe9b814344c777ef`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff276f984ca7449fbf40ed52f8c73e2aedf05be900e45cdc0a8a0b8a46668591 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c68278c6557fca32e7992f0f7223b1f9b76c50fde29e83f1a6a475094213024e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-212fcfa9285ca599d34ca6bbe7ad476a681de6b25b4166fcc7d4b200c8243bbe |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-de21025c9689b9986318f68e704e8925076f21d0a49de86ba2b17d5f292f3aa6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-322ffd40 | sha256-7145a6ff05855092d0dd372ad2828ccdfe1663583e855eb8144943386026d0c3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-322ffd40 | sha256-65458e646984fd1fbcb06f827286fe1960d26582ea3d3729e0759c7d287b56c5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-43488be5 | sha256-bea45cf0771751ab248287c6b9bcb734ed4e1246ee5222606401ec9bd92b43bf |
