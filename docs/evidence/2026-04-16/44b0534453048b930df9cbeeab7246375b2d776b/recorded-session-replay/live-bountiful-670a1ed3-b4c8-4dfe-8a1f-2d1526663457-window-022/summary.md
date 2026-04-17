# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f9aa9aeb2a008ffbbef66937498f659450de790103271a3013e9525a14c6fe94`
- fixture hash: `sha256-5a27682864273526a5ef1ec747be28d22cb7ff7f18b59d5b0629943c5f759e11`
- score hash: `sha256-60555b7cf4b90783475b5eab9af48f97174dd358e2df8f373870cb3fc97b33d2`
- bundle hash: `sha256-4182c07a4deeede369f79aa3dcee4061e92f48c10bda50073a729628124cb1ce`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-739b030030884c96251dfe3ceafae4a4f46a19e7b6a4299475a7d110d8c8623b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-27410dcb1351bf8ccd5ceffd2c2428ab004c868b0596f31926c19098d2734478 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-fd60edec3960cd77ff08424e0d0639e045f5e499d69028663e32a84d8381639f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-db10ebba | sha256-81a74e8afdf5149c96fe0519b714650838c31c9b81fa5d614e21a342eef840e8 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-db10ebba | sha256-a4c3109f253b531f5f19d6414f8491cb087e15569f1564ff5c0353b9c35dcd2b |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-4011a54f | sha256-eaefbca2f254269ee09a09e2fd71630b3ae571b22863e511b3f4b401fd2d5825 |
