# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009`
- winner mode: `vector_only`
- trace hash: `sha256-1eeac1e61d3b2146227988d62e1fe6c84e0b77c1468137fa8c6d382736c2c4ff`
- fixture hash: `sha256-1569517b028e54a6250341eadd5d277f396164c98c963a62234840e80af05420`
- score hash: `sha256-9291f05ae4fd0b153e0d3d9c87a6aab07011350890919064de6aa18e898a6142`
- bundle hash: `sha256-4ec9b9f13b527a92628ba88b2e602b3fde570dc74b09520eb2210c5888134b32`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f9c2cda00f69f271539f73d1199042ddf0b5deafd10dc5b6f5c335b4828960b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e3e62a16c9633cf78d0d4f5b697ef806ab3dffb0fbdaefb1b564dc768b0da4f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c1f2de0fcd0f0885ad62b41c94d990ee1c8afb7c6e6110e0d24fb93a0403ae89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9b14ec51 | sha256-f48b440fb06cb2bbb48d6895664a886bc59ef90aef72685581868673425242b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9b14ec51 | sha256-eb9ea7043d5b102cddc9f2dd6f1991406a9f4492dfa8926b017e459713b0a923 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0db0253e | sha256-924c3c7ab042347da8fd660dd7c3cec2707f6aa0a0ff63606f81e61046490cc7 |
