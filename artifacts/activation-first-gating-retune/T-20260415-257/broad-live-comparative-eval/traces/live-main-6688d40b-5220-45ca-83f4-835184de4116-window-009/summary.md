# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009`
- winner mode: `vector_only`
- trace hash: `sha256-1eeac1e61d3b2146227988d62e1fe6c84e0b77c1468137fa8c6d382736c2c4ff`
- fixture hash: `sha256-1569517b028e54a6250341eadd5d277f396164c98c963a62234840e80af05420`
- score hash: `sha256-8f48359130a506f3bfcc3b986f86975d0d20b0c5994f54f9aa6d74935ba571e2`
- bundle hash: `sha256-bab9812d07022c6f9e8fb7b6ed06376d3ef7a92657caf5dd30c9b6f63f7e776f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 60 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5439b064337936d23e3ce0669d08085a2f0dcec2b235478161f6d9e74cb033a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-67b6eca959977da5a5dd7851064fa2225de0ad672a09511e59b5e5bbc0a8feba |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a2c37ac12b9a182de3f7311522a1d27242258227a7294defec97d2b305446822 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6295cf8a83ceb85b67d1a2d0f04e3692af5f7d0537691a71176d9bddb8877dac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-7d52f332 | sha256-38135fdaa19dc77054dedf69364e554e97850d50f187dc1eb8897f0b522aa68f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7d52f332 | sha256-8c38f6099b8bcb6bb79b7c6aa8db18c8b641849e55ebdc59e47b57e7229d229a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-efee2c1f | sha256-c786b5a58168ded9ad341cd631055a2970978dda205bb46f087ec3522fa97e28 |
