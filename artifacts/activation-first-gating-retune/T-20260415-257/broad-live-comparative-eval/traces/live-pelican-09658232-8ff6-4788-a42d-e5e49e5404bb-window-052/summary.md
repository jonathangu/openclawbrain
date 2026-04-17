# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eb4ce4c16a0b4086f9bf16153627317bccc66c138e3f3eabb740de5aad356d3c`
- fixture hash: `sha256-f8df6b8b0d3896e4d68df7e66273fb59a221dba4842848c8bd3431e1201171eb`
- score hash: `sha256-3caa203bf5e2b8a85d4dc84744e7cf9de8d59b22b5e20fae81aa4340259ccabb`
- bundle hash: `sha256-104dd47b6197d242790668a46fafb3fcf7bc68b359dfb249f8a008a4108a9adc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31a831566ed58685dea8a0c35a91e51999c06d52779d7820057deddb5dbf99cd |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d1d1a5d627894e69567ecb5185dd9ea6736cfd3193b63230a6415e9c7bdde5ea |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-be94aae00c7e3a661040ed1b4e48194fe759cd115f2a89ee792bc47a87c0439e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-90dde3f4f17b307521cd68021b74637f39e40447934c510b4deab10edd81f811 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-099d9005 | sha256-3e5e5302eee62884baaa4e59d3bde4c2111e6c0522354e60bcc44161a901617f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-099d9005 | sha256-5877170b637f1832008c21ecb22a5b255316af96a39aa869ad1082daa06a2429 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a0c2cdb0 | sha256-91a7aa48ef7850640d9ecb48d6efac74c9237f66376f710e5550f63868c66a6f |
