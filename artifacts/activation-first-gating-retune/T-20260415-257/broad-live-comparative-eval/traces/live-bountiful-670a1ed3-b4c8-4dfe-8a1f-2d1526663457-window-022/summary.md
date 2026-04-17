# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f9aa9aeb2a008ffbbef66937498f659450de790103271a3013e9525a14c6fe94`
- fixture hash: `sha256-5a27682864273526a5ef1ec747be28d22cb7ff7f18b59d5b0629943c5f759e11`
- score hash: `sha256-ce2fc023fc97635793f9c25d681d1d6055f21a50d5d3241c2e35c1936fd3ede5`
- bundle hash: `sha256-bbd0a009906bbdf2c9766ad31c8ceb68958df48fe5680170b4451bbbc73aa906`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b2d51f37ef17f0ed82a2f36897126b205c47228efe0e37855cb029004034490 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d880353e130493ab5aa36ecc3a92aa53c9941f3e8f91801906dba77b4b324ed1 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c8a1e5aa29ff16ffc12f4bb1d27c2b06dd5c3c63db8b0e8ce6c4a9f3b18dc035 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5e3265abc6f918766c46cbcc5544c057149b5b620397a785722f13c11ea8079e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f93b0d48 | sha256-fdd9a1d821ca2933bad704b97b0ec37a54c6bdec1aeff61ed75190596958ccf9 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f93b0d48 | sha256-c0f8dad91c1a360274a74ba5b797ee3f8ccd50b429b62fc73f8de202c0bda23d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5e3bc6dd | sha256-2e2fe985a1bbf4b24d00e6d8fbdbae5749b4709aa84f54c2673e19e169e69be3 |
