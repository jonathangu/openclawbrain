# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c07d86727cebfe369bee33466c114948863aec275c3915842adcc6210ff9f00`
- fixture hash: `sha256-8b304aedcdacbc80bf121116c28b99b2494a777738f36a03524f91c39297ceda`
- score hash: `sha256-41d02567010aa77534847e5e4a92cc62b1aa1bffa9cb6b84b8f8eb701a51014a`
- bundle hash: `sha256-86dd409f6d65543c79e1794de42ddabaa1a375356b126108091677d4894f087d`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61200abce9d599bb6b2839cc09f35d3da44db5dcfc15754c19ad25f67b630577 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b4580814fa844161dce966ca97ded50468c5047d8204f354ddcdd708415e8e6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e9eaac04a15b58275040f46e10b00dc0710790fc7997a9b2a759ffd4016be12d |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-eeee55537f4727e6af1c9ad7b5f709351650090ddb29cbec927229bcd557afe3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-88c1d600 | sha256-4f43aa73830d754d8308a715a97c3d6c4d8a6b713a3000940bf3031548e3c87b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-88c1d600 | sha256-f8b4ab5627b2c5e01dd65f5730717be72eac611929a85c93e2dc1a92a7b66c58 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-88c1d600 | sha256-4f43aa73830d754d8308a715a97c3d6c4d8a6b713a3000940bf3031548e3c87b |
