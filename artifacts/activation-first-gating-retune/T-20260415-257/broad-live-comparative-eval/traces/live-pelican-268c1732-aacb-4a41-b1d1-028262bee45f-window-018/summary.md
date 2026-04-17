# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97d7d39c8ed80340fd41820d6d636bdacbec2fc0c19c6596d376217775b20481`
- fixture hash: `sha256-cee22d0c8692c9c54ea684f49e1d3ac5076518c4157aff7a2d52bb3e3278c63c`
- score hash: `sha256-e444e5e4bf84f8728d298d1210291b154c8b8256315b21accd9abdaa726491a8`
- bundle hash: `sha256-350c5d1a212cbe067dfb5045b5dab0661cc09c2ddc7be7ef74e295e93ec115b3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37de16724e3f909b52770a9de834272378dcc6d8dc93db3d2e32057318f060c6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17997fe11aba2152550f5419092c17652b5d23563b15181f16705bd47715ce8f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8887b455b629812fc68168ec83604765bb73b60dc641bcefe92a19add125fb8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c19756a61cc41dcab9d5e4a1149c85c2ebea00ed55a1f0f7715fd32e80b70409 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-831a203c | sha256-90747315605897f28edbdc2abbafb540d3e1453b5849c49f09bc23a0611846d0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-831a203c | sha256-64f44a472cc1fd15317aee70d617ec5306ed13a754a121391e3f681c6f17de8d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-97616b81 | sha256-6b4dbe647222811a8b9c8534a5fcd7f58736608aca9c9d9b1d2e068f9c1dd2e1 |
