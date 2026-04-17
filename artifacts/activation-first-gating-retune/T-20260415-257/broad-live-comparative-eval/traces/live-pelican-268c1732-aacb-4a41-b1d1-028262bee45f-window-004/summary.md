# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-0cb637b60338a4ab72e814fc55249159c894cfce0665c28022a49005c8e06863`
- bundle hash: `sha256-be3529533127908a9bc3a6112daafd967a6e92dad47eaf32d9257a3801c9da8b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-964c03743652d9f70ebc684b505f2849fe0fcd1f24bb40334170405c3a06d40e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bed2f3de3227d7b0bbb19b382f36efa522ebd46f08dd0155d0024588bbd081b7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7746fe09953855878b3a4566f682ee5d0a257b60883f9696052a9900eba9fd49 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-966bb923 | sha256-91e3c9daad78cf74b7b454aa270286e08cf6e92dc7e919f0dd42c9ceb4515cc4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-966bb923 | sha256-d39502bbb55c496de2d88e972a3487941270b9b7ebd8b274a426cd9724c2686b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cfe07ebe | sha256-342055ba0166d5317f2066c806b64639f547da9b08855097e9333a1550ed4e47 |
