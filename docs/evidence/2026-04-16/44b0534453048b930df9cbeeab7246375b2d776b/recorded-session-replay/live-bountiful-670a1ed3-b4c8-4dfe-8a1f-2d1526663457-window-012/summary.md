# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-14ade459ea986baa6e4e71bbbde0e89dc1fae7980400ac765d36815dff4c4f35`
- fixture hash: `sha256-9c30c978d165bf9a25e14aa9b77d9a12a45f7a9014b4a8204bd05ec1ae139d4a`
- score hash: `sha256-ac8038cc19067622fe4ad6b3df92acc38f9451c4d70bc256e5b5c67db45c7bcd`
- bundle hash: `sha256-df0673383542a98047cd4c58e0b27951db022a1007e8dc567adcdac1cb2c902b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-334c135bfa30ec156738872f694abf9297995f829f0e8e1c5041f315be0a98b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa0b86b547b17f50a3a63e7d0424b024da4de1dd5d758d3f893e5b7bb9666de8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d0b6f19d7527e821669eeb58eae10ae55359a90178ce275898c607a7f44101e0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7e36cd76a1ea0ce631aefeac096ee1fcbd459df133e0a2e9ad1a564194bc4a9c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dc41f56 | sha256-fd9007bf3ce551386fb5d94210f0f60bb5514c785647e752211bfd9cac1373c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dc41f56 | sha256-6539bb1425fd0f5bec5ef9d240248a30997286a80387058083f0ecdde3b7298d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7d9a0bd9 | sha256-994cb4d897a3871419bb3f275e22e87e4b9add299280ee474665f102332bf55c |
