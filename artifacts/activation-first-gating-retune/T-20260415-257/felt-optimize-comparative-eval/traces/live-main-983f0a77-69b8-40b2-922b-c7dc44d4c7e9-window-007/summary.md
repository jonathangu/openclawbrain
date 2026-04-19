# Recorded Session Replay Proof Bundle

- trace id: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13518408454d88b3ad692b956343d851ffe682724dcc9ea68679835cb38cd6f1`
- fixture hash: `sha256-d8ddfc141ca061b024a7735fc1bd6c41a09ad3c89f85b7541ee5a4463459f049`
- score hash: `sha256-6da7d7607ebf5c8d3fcb7a896b03b872f13f2d97dd27e747586d79b370443bb0`
- bundle hash: `sha256-24649f8c1c0559c8833731003207ccf5ba6d2151c45d98b1f02916581a397dc2`

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
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-493e4faf92188d489cfb2e18ce82561ec2af21aae9c05bf198b81e509829f171 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-7fa35e626797080a0795ff173334b817297610182356f0982ba79dd99be4a710 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a1dd16a96d0e86383c78cedfcfb43557fc55290193c42a78a8ee50de68b10294 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-40bf5c19 | sha256-a644b71af427b3920d5031ea11891cef462361ab2c3d55789015181140ed9d52 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-40bf5c19 | sha256-ad3fd64ccddc99272687cc6d99d897b98131ef564c9ebe8d2cd0e71c986a1cb5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-40bf5c19 | sha256-0c583cae0e6787d55ce36ae95bb26ec8fd0ed3ef97d1309712d0d9fded0e59ad |
