# Recorded Session Replay Proof Bundle

- trace id: `live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f45e438e0f99f9d56b4e0ce3ef341383c4f2368651efec5583a2c7447c8a5e0`
- fixture hash: `sha256-24e221e1cec238f614a332fafbde124000574c7f4eca983f394d512d73646f16`
- score hash: `sha256-e6d06d4edad4e4f2f23b03be5f227b2550e7336d5aa06157907ad87045403b6d`
- bundle hash: `sha256-8f0ebdeb52d9fbdaee48662468808fa39baaf9a082431607e2913c70f907d178`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5624a20b92c6a5c4c5d269dbfed46d621fb3009b7407cdb61d3d2abad216a892 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d7a22a70785ac3096aa081495a40f41f64f48a04289c9e564a927251420e7a99 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-364f524885f333bab13d3d79c9c8cfea6a7aa14a133036f3ec5c732eae373d62 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7c4275e7a79480bf97213bf1d1fe275bc4cdcddd9d9a15b3e002195c7591b4b9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d91ac828 | sha256-0df3fb9e91ba2e996719fbdb4ec24afee6707eef424b3fd43a51c7b42b1ca1d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d91ac828 | sha256-169315c53c76e35c25f4b22fad19613d2ce952ef640fe9bee35697105e387b2c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c6a44e77 | sha256-7666215fd750727356d822475b0cc85a017cd433d52c7e18722f42f8a233b6d7 |
