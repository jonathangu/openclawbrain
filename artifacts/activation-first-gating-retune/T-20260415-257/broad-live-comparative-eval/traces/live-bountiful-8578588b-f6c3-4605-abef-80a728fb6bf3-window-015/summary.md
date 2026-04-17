# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80a26a379e19395f27acf926c2152bf691413288f74fb851622ffb175287659b`
- fixture hash: `sha256-b5c7f982a4c7c837b7b862133d2b0fb112246d1e5da8e79088a56147358f56fa`
- score hash: `sha256-d19916ef834632750c22f8407a982243af2f00b3be9ee9f875a602c078073a6c`
- bundle hash: `sha256-39fb001da76e52daf0331d40c54183b04177a000a4dc2dc914a1b305d00be226`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e03ec51b785f9aa9cb42beeeff27828af175efefc4b17a7390a8051e0981fb1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee0deb44de08847aa00fc809b2daa5a4f938c0ddb4a936d617339202ea1e267f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-47ac05dd6d74f3fdeba14f1fa0e62d62c5bafeaffff6b27dd4a25767c91ca661 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0bb64221c77fd5979686cfad5816770fb3e67d3de22268042103b832af4037a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d9564638 | sha256-958149080c4bb3504d13c202689e35736157fbe7bd42ac39d9b029798ca2a9e5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d9564638 | sha256-b1ec33beb8a3027737b10f859c22f7375b8a9b0628179f9400b40df4b04b1ca7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-469572b3 | sha256-1166a9c77b1016a9a08d29f52002119bcd8154838da62b84c809fe093358883a |
