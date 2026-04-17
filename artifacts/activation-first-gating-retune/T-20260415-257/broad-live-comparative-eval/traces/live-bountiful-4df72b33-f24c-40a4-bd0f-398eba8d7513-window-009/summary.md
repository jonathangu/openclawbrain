# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-909da7829d18c7b4060630b0679e4b4f9d3623ad878ab660055447ba4071489f`
- fixture hash: `sha256-d6f02df1f7fd44472c7a5dc57cc2a6eccb52d15bdd1b99973bade05111901191`
- score hash: `sha256-0fe456715bc8e27ac989785e426723beca5a539d363eebe0df12bc1aa7898200`
- bundle hash: `sha256-a6d3fda2626f278d772c8ae2d3161889bee305ec0a22981bc04ccf6e5913839b`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2130d4f9265322c2c59cdf7ba78394592636670969b94ec60fd50b3944919890 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3550f244d0bfcaf6d81250529f56037ff88a18bc695cbc199184686a11648d68 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-55c10ce4cd70bd09035ab5de8ce9d4c025ce8af6869ef9f3e31a56b676d268fd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-25c31ee4 | sha256-7597f80d7832e930e53b454dcca8b6118d2b032ee47705a799cc88bcd11ee5ed |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-25c31ee4 | sha256-5bc5baac9982fb44830d882f3c97713b96091b4720ed75659588ec814fec72cd |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-beb7c6f9 | sha256-44a17eb47135295b34c0c195b9eaed26a5a042d8b55395633cd17e28024c8f13 |
