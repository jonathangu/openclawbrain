# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e6da83215f56050459d7b523ac12aa0af75a3f0c2a58f526978f2254a29ebc94`
- fixture hash: `sha256-d4387ac5a22395546761e4051b3bacd61069a298c0e71126f4fabbe9ecc70ac1`
- score hash: `sha256-a6bcd43aa9b0a90bd52ef1dc4d376694bf7d00fa452ee474806ed478dc601242`
- bundle hash: `sha256-f761457b7a523ef9e195d42fbdee6ab1b0deeb3c8f497fb7644f71ca8a8d6a1c`

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
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a588b2ee957ec985d426ad6e49fbf57081b897d4927c49e13bf488a2002d7bc0 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-c6964694496b6607708cbd4568f5847b2e8830ec9fa50464af28836af36e0107 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-24f9ed59523026eae9d750e9fec469fdc70e1b968166a8a88b24fac4e9fcdfc6 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-e4ef52e13183b8992adcb8112e1b0df2d023fdde541e273cd080e9acbaf70e68 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-9e144a0b | sha256-bcbb473284de30ff84ab575beb7433a18ff1b301e554c944e521d572785fb785 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-9e144a0b | sha256-c48f15c0644feea32efdf0d9b71edc9d414431b2359ac44f6a3e93f5b56922c7 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-9e144a0b | sha256-bcbb473284de30ff84ab575beb7433a18ff1b301e554c944e521d572785fb785 |
