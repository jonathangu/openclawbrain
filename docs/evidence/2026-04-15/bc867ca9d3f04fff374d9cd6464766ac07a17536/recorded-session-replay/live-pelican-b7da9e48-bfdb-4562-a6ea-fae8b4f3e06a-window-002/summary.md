# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adc21e40f3c3bdc2111e458183ef292b9fdba4cc9072a5e4575150e3a25e7599`
- fixture hash: `sha256-82594518eb539bcd92075469119fdd7049793972cdce0d3d047ffdabe9e539b7`
- score hash: `sha256-c991d61d83a1de8c199cbdfabbc654519f4e2ead44b0aa78e647a6d062c4f9d0`
- bundle hash: `sha256-1246ae7669b693bae19b44d75c5791f946eef71c9b0e6a26e81e255d59b40b1c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca48aec6e03fc6ebf10d02ee2af1729bb6ff692653b0f22ac3e3b10f844865d0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8bddb14cd49e05c28217d3730ffd46888dccb381ddf540f72d62f4ddbb5184a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d04cfa7d4b8a58bd09f8e3664eee8bcf1ae6f946ca1281af5482115bbdccc311 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4e95976441133c9a28b703716efed84dcde7a34e297d74a31cf0d519d22d447d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c8bb95f0 | sha256-512cb10e5e501911cfac27aadd321f8b9af402645ddfd6fe196864b04ec1e748 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c8bb95f0 | sha256-00ac46bf5f84e567841891675eeb96c0f22ec5b6e62cb44cb8ff07b36235da8e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c8bb95f0 | sha256-512cb10e5e501911cfac27aadd321f8b9af402645ddfd6fe196864b04ec1e748 |
