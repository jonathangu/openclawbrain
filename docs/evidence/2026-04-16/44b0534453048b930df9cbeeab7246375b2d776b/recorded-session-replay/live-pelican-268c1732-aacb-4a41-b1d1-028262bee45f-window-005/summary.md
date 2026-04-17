# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1d9740289fc2adbace7590e78dff24d1d94c6a419d6474e3af27754996da05a`
- fixture hash: `sha256-b9420b72d3a2c2c9c62adbc0b7f3ef24407bf200cf73b9c382cce44e2d33fe6a`
- score hash: `sha256-8c8a279d6f0603efa4088893e1256309cf742dfd49a03f4189c645f2a6522bca`
- bundle hash: `sha256-77cf9c44f2f780f194b80be166ab392ca017a4a1f29de5204272db2de2a7a4ef`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36b1da37ad1b5e7a8a6bf9a89b082e6da9affb9cefe62c4630aaf0bc52cbd76 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-314b56d71cf614cea1cb03acb1132859aba8ef892a1bdeae0df0fec7b8d01ab9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cede28e0d3f075a7175f15108502f97d4d4cbe7f176df5d6f9efbce348ba287d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-bdddf40bb5e29540350240f805ccac5c5ce77062a908bad137a82ccd209c96b5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-38990803 | sha256-c36a9906c5b37c0f8474cd59ea25711ff146962d82e4d01c2cf45eabf6303ed3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-38990803 | sha256-2b0627dcc136e3e4358e560e8758e2b930a425c090c1ed75f195a153cdd60120 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2814a36a | sha256-1f4528cadf786cf2e7399bc7c870474abe6f8b0db222f87ddbde59db2bb33891 |
