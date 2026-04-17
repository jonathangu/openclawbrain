# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-34a774cb3f6c8a06b7737a6a2929058386a540d4a4f6fa06d56dab519cbae33c`
- fixture hash: `sha256-38158baa488957f4efebe2494068936f86320ad50d0d4566b804a6468d20bab5`
- score hash: `sha256-7219314450ed302c7f38df4f1e8d964bc0dae161ea8d1266c5c6522fabc10b13`
- bundle hash: `sha256-84124b50c144496042c5145376b095d37ee44a05bbdbe44b17ae04461a5bb10d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 60 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2a941461c5687ced5f6be63f00e8602b946e4d86dfa5dfb8e215a577d1b9170 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b603b4da6f46490321f73648291a7682d7c3fd97afc597b45766c4041aadb7fc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2723005a6251e138b42493fbef580314a4850a1aee11b6aa7d8773d76fe048a5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5a173d0110091987453ded3993c3059181caa1b2eb5ac70d0f442b6e025f7889 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d93ec12f | sha256-cda8d90364bfe55b4b8060dd46939f15dd8379c765dfc34b652c983ebbcab0a1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d93ec12f | sha256-57b04b1b2887ad79a33601a720a4437f3098677a013092b9b6ac240df1f23bd1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-81ee5346 | sha256-358ae85460b506e562acb5f61580a88cab3a1c64fff47620e0a3500c1eba73c4 |
