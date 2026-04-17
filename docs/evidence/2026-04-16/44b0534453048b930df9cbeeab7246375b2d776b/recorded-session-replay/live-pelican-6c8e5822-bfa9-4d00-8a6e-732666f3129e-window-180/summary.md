# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e44bd492f128c06a27f22e67cd820199254d2a3ed0a6ac13485df4261f57fa9b`
- fixture hash: `sha256-cffeca9e647d7d047b9dbfa0c2bd2eddc1a7b9897467d5e861f95728aa0ee6bc`
- score hash: `sha256-410ef421fb63a75c3b061d07a8f3a5b7f1bfb01f2185dfe8d6eca885b902a964`
- bundle hash: `sha256-2622cfe73fe91c92716a8d1d5bdc400fcb5d4c0448e370c203fd55534c2dc9a0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7211fa1ebf40e19b79ecf69c6d2f4cdaac759ca9e3451e680c32982ba6c5891c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d586b6f3b73e79f8ff8f11baf5176267cccff7c8330cd207f6b2069699a2e779 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-588b1a8954ddab09ad5ac7e7d587ec6655aa9426704c482ab8b8aa08347d2e5e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c03a172e72a1991a8acea02f3b877d9b71d6a8331c4a578347b9116418ded7b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2bcf5a18 | sha256-fb1855bf06bd95fcaa0046d4e3d811c73c0068073035e04d2bb4dc9ecb0b5067 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2bcf5a18 | sha256-153219f7a31da932aaa9096325f49b2496675d643bb6bba75e04cf20170d863e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-19c404df | sha256-04eea0bcb4dc3242721a1e259a8cb53debb10ff64629de9aaa1dfe4aba791472 |
