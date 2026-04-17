# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149`
- winner mode: `graph_prior_only`
- trace hash: `sha256-299806353ab465c5dc0556cb46d4c0ddab82caef7c74016e1f229b80f14988f5`
- fixture hash: `sha256-6a01dd18700c95a1ef47fa69bd96f40af05494c628d07e9816bb8fa24129ae15`
- score hash: `sha256-4e06c4c5054e7ea5f4ce94c27ab2f3e5ec7dbb7583f66f40b1eb150f26939bd7`
- bundle hash: `sha256-065c483d2f8d3829fc477f180daf14e0199f56f289f4f1941e193c478e6e0ea5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2b5dc7e86f33e7f35222bc8995c5714891af6e48c0b188589cfd85f30ab7cd |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2810f80df7474a9c4a1e04ae4230e14f318ba0d8321773e9176931272d0ab9bc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-856341e848e0a8f699ead06a184a8b62d920111ad6eb40a29d0c00d39a940d43 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e871fcb6848de5eb4f913832f5790719a1dba6fe30b9a194d7b94450d8a88496 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-73adc31c | sha256-90508031dcb2890102aa023bd2467f982b4b41909d9e88863d12940f30968aad |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-73adc31c | sha256-c2aee38cf1803e7a834866317980a39e67191a3cb8a1371115e0101ce13f37a0 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-198ac289 | sha256-acb5b12c8da65d0ae0846a20927d5bb6fcb229f4293237f98bc7c4518ec2afae |
