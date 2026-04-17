# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f337daaeed6bc47fc68765c0195f530bb4ce38ec076e00ac4c73412b426d85da`
- fixture hash: `sha256-5e5cab708ce5b294bac69d34a6279b47e648ad8d40ed85f35998caca6e589c7b`
- score hash: `sha256-bd0729f4d8ef3e5fb552a6f78623c69f0bd18293589b219f43a5294efbfd2c1d`
- bundle hash: `sha256-ed7cb90c6cfc6338faad2264cdb726c8f7d7224feb92eef9ca3665f215086cab`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-100c819b269094add6922ee0aca0d157fd41366c476c3703f8d276f1431d3315 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-676b009a5c5b1a32d15c87360464674d3574dde3f708161a22148f6ff229c10c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-77fa1f0502803e1fd34905d7617c8218ea41f76d5f8cd9c5ef243662a3534094 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-47d148a9dbad82268ecdc1e27790c2fa1748056f461e92576bcaf92fcd5cbeb7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-07607d4e | sha256-ea629ae82be7db45c99a14e8940189dc820315461dc3f8a61db5b968fa556fc9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-07607d4e | sha256-95b97519d4ea3be0fe364a2cf6f9e634298c163b5fb40ddfda2d281dc7688fdd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-92099e5d | sha256-afe4f87876b708afe52e5f6c00dcac2fb4f783ea138807f84fdd887adbba5d8c |
