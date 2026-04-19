# Recorded Session Replay Proof Bundle

- trace id: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13518408454d88b3ad692b956343d851ffe682724dcc9ea68679835cb38cd6f1`
- fixture hash: `sha256-d8ddfc141ca061b024a7735fc1bd6c41a09ad3c89f85b7541ee5a4463459f049`
- score hash: `sha256-bad935ead26847c22196a8ddffbc5c8f5f47d14cbe50364f564aba2dfe5feca9`
- bundle hash: `sha256-9812b74d79d5194b9070cb4be0c3ee7b7557f97398bf3ee8a32efefee6373f9d`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76a9870f23308038c7dfa2834df546254ae4769b20da16b32ac7e7ef5f9b078e |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-67ec19c8eae1e25f35ef8a906e3411e6b7b57f69459d0698ab59fec3bab59b44 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-099de546fb3e47f846de17190d8f9d258dfea7a939d2bc5999f9f5eded1e09a1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9944a8cdd3cdd8c599ac453a4f52bc861203b39a9a5e130ea82b85b6b200b865 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e54dc074 | sha256-69d798abcb63f0e4e71bb7c45c277b8c777be4f8126b1489b5e2aca62d4d4dd7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-e54dc074 | sha256-e24a36a15241a48dfaabc8cefe9a58ddbb42d92e3da3791635238a14df5fc55a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e54dc074 | sha256-2c67aa263e614ddd4ab09860c482ac6e395a7f47d0d32380c18d3c15fe1d5e10 |
