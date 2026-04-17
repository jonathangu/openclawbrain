# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-909da7829d18c7b4060630b0679e4b4f9d3623ad878ab660055447ba4071489f`
- fixture hash: `sha256-d6f02df1f7fd44472c7a5dc57cc2a6eccb52d15bdd1b99973bade05111901191`
- score hash: `sha256-a2a8ca1c206221a2a4185966dba67c0e4f636555b91067b23ec47bf8c8dfb67f`
- bundle hash: `sha256-c4f575391ceaa252adc815d6125eb97e09a5a71e4d379fe53501f38972bc8519`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-67d503fa6cc9652ac016339af3a8c1038900f3a66d390047a5f53eeceb6dc18e |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-09680ddb16dbf4e739ff8ecbcdb1207faaa7ce65eba98892f2c224252a210f4e |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-cef17b16bb6b9aa38858a75060c39c4b0c390bc677ff4d46e188901a83caf6ea |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-308a6e29c6534cd8bc297aba7a63452ac6e432c3fb543507c47a7d2c04677ecf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5a2ba37f | sha256-6ae6197ee1ef5d18cc272e0da30b60dc310920cae46bb6990a564acd2ca75fcd |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5a2ba37f | sha256-42bf7aa4c5e0b193d96eb65c886746f752c6d2ee5b509bbd4b7ae14974f1cdc4 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-f3204b94 | sha256-960ceafcb06f6fbb41aa5687ab8e1c6a37ce2942b94edfd546749db96ca3b390 |
