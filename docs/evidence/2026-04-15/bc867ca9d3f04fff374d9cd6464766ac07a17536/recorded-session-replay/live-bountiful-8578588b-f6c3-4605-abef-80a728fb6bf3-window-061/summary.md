# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a462611820be4f93d663b55c36826ca6c875b16681bbbac07441c4d98461cd86`
- fixture hash: `sha256-55dd2817c613b4842f9e8a859b558557a568175128bc3b05ebca7185c8b4c45a`
- score hash: `sha256-f4a3a9abe49e4365ab0275e503b196592b981f857850f5344167c5818d4e2c17`
- bundle hash: `sha256-a1e73c519572a838eef86b13b0e9cf1293fcf103569cba2ce89140ce9fdd5773`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-74198a04279dede09056c15d496cb33d205dd223adf4c279bc3faec0cb8bc3dd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-955f7a7e1a512aaae712b0d91e3fc84fa227fa017dd7f45d1b81e1369608dbf6 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-da05ea981e5c9b7ab206cfb90b64980b7c2cb10dfdb89d2859f3c3e77d000f36 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-9b338d7765ccd1dec574dbee835b8acdc6b3043ae77e97d4b89b0ca2bed3c6aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3f28306c | sha256-909afd23cfa0108fa895fd0e712aea2fbaa295ce9a49bb7e83ffa2828afb2e83 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3f28306c | sha256-dc59d31486a4636c3535b9e04102913571147bebb9db618dbfd6cd4a18e8ff36 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-3f28306c | sha256-909afd23cfa0108fa895fd0e712aea2fbaa295ce9a49bb7e83ffa2828afb2e83 |
