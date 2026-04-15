# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70ab1ae3977bf5b8f105672a2af7f511f5d5e8eab54227af0f4c11c32810b91e`
- fixture hash: `sha256-f43d8483c3b4eb473890c9d4aad38b8eb4a81081d719d9c58fd2752db7997c33`
- score hash: `sha256-193d2bd3ad97f75715b23c12e3029699934710e27caaa3413423b3416d1e00d9`
- bundle hash: `sha256-c0b4a2378845d496ac6cfcb72f4a9a62b51fc22de735d9d2799ec015288d76b8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0af8593686f4dcd1625a4737259415fed87f48af0fee073ee2e87cde2bfd51e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0eaeb26f644dbc1e28a48a93f59bbb09b08ebff3690b4fce455c9b6ff26a80e0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2d8327a70a15dc4d8eb058ce2783de72529ac6aeea168a2b059fb09f7261c9f4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f4458442eba3842be6996d93666c7b21f33f8b3f025d8ef90a93d43c228b53ba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4adc190c | sha256-2a1e9acc4eb350389c1c919eda40bcb486415926c7cc66e56596f4b756a22775 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4adc190c | sha256-e380f2a4d997beb400ea23f7da3ba830e496b6879d135098e821adff0e2dfff2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4adc190c | sha256-2a1e9acc4eb350389c1c919eda40bcb486415926c7cc66e56596f4b756a22775 |
