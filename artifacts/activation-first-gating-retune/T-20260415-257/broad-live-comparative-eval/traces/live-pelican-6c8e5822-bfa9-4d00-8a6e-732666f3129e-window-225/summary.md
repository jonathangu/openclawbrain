# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-225`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7768c82b82cfc9e79b11d2862950b229d848ebccb180ebcb1860140cf56b1f18`
- fixture hash: `sha256-e3d4346656fea9fcd52a8093d89ccf43c79e719fec02594aace8851b57c7f190`
- score hash: `sha256-3c7d9da742cc36ce65360154661234c9d0797dfe04896f8933d3d58eda735260`
- bundle hash: `sha256-5cfa82cf4115367f1dcfe9b612bb819c6ed7f1d783cd83a9a484f642ac63017d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb4473107ca7b170cb7198e9a132dfc26d383b8a4567d404be160b76d2d08390 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09cd5baf8660d8909c0c92efbe88361c24389c94895d18ebb3798bb015089b80 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b25c576d91e4bf48710eb543d6e99e28ec34cfa222cacbe83f9023d096727fa8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cc84fa1578ead80fdbcb14bd46c82edfee6c185204affd691d7738dfe9581f18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60164c02 | sha256-2c94316f2a40466fc3b8d0eaaff375e31faea9588d9b64664f9939c762d476d2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60164c02 | sha256-0fb4ede08c325af2b8c56ffb6d0d6b206425fa53bd57bf0e9cbd012a70d3346e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-da1310f5 | sha256-50a39cb221814d6c8ad2b4a1673c31f4ea50aa851f1807136da85d926d7e28ef |
