# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60561f5cd4b9679d1d07ec70fb93c8ce09ef36cd5a40b0352b67931141e9e246`
- fixture hash: `sha256-d68e77ff5a53346b0fae859928eb6131851ab9f7d88f52a94509c0f85b109391`
- score hash: `sha256-41e26a82714943775f99fbdb997de4b87104465caf21070f93281ba6f6aac4ea`
- bundle hash: `sha256-ac1a25533f1ee0ca052dd7bbda0d1d5d5f6b9c19362de6a9062cf3702438074d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-350a961987884e875cb36c29ae1cb810ef961abe38158c92bab3e2c95369cbcc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b702253723dcc0dd175c3419a81e73b6a1658bdf351bad7a75155974a5493a5a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aafdee512d81bc51005a8ed3167319f4df852a4dedd8cf135345336c3a6cc1d0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-89c81ea533e9e52694b912f7ca6e2a8ea4d3298c6a7690e550d9ca97065114ca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ade64e87 | sha256-1b32643a40401ed1fbefb3494568fc774f0e6def5a38e00af92cadd9b14248f3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ade64e87 | sha256-fd064d40d61cc631cdfc1b1c3845316f4150a63c68f54c846ccb889539c934de |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-05de8fd4 | sha256-326b5edb6b8adcc08f11de87d58c8dfd003cebef2179c8c44bd0a0e9ce60ab16 |
