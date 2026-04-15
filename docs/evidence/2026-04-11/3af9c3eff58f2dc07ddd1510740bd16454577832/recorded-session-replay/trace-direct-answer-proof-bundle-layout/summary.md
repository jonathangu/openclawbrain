# Recorded Session Replay Proof Bundle

- trace id: `trace-direct-answer-proof-bundle-layout`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54052d000bc2de2ee0ee13e66c958eb9a0908be2b39a2b95467eadc464a02dd8`
- fixture hash: `sha256-ee85abc4e14b2b91215595b14426e9eacc4b89e75df8b01c0a46744b81127aae`
- score hash: `sha256-8118d2f44f248692a5a5b03f07247c8e357738ebb8f7e02d81ae98aeff1760cb`
- bundle hash: `sha256-2565d2a61498d864cba5a074e766b17f0aa4c119191761d60fb2e2ffaa84134c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 88 |
| 2 | learned_route | 88 |
| 3 | vector_only | 88 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 12/20
- phrase hit rate: 0.6

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 0.8 | 0 | 1 |
| graph_prior_only | 2 | 1 | 0.8 | 0 | 1 |
| learned_route | 2 | 1 | 0.8 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-07e23608e09214276f8ae4f5271602de4245cc9330c6f83aafcab4e29dab8b2c |
| vector_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-4c8ec0d8431e4a5402ad232b37d72d5087472cef565589a9771488a10c96e3ab |
| graph_prior_only | 2 | 2 | 4/5 | 0 | 0 | 2 | 1 | 0 | sha256-4dc134a5cb4c8aceac67b3bb1173e9d67f0a5e6b72b619f1eeb589b9a7727f94 |
| learned_route | 2 | 2 | 4/5 | 1 | 1 | 2 | 1 | 0 | sha256-440e46c658416efe6360be2ebdc51135d6c8ad706edefa5c7966addbe7432dd6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | bundle-layout-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | bundle-layout-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | bundle-layout-turn-1 | 100 | yes | 2/2 | no | no | pack-6efee4de | sha256-60572ca5725a44bba621bc296da8a03d180e8fcd085c5e37d2466eb870034f79 |
| vector_only | bundle-layout-turn-2 | 80 | yes | 2/3 | no | no | pack-6efee4de | sha256-60572ca5725a44bba621bc296da8a03d180e8fcd085c5e37d2466eb870034f79 |
| graph_prior_only | bundle-layout-turn-1 | 100 | yes | 2/2 | no | no | pack-6efee4de | sha256-60572ca5725a44bba621bc296da8a03d180e8fcd085c5e37d2466eb870034f79 |
| graph_prior_only | bundle-layout-turn-2 | 80 | yes | 2/3 | no | no | pack-6efee4de | sha256-60572ca5725a44bba621bc296da8a03d180e8fcd085c5e37d2466eb870034f79 |
| learned_route | bundle-layout-turn-1 | 100 | yes | 2/2 | no | yes | pack-6efee4de | sha256-60572ca5725a44bba621bc296da8a03d180e8fcd085c5e37d2466eb870034f79 |
| learned_route | bundle-layout-turn-2 | 80 | yes | 2/3 | yes | no | pack-029eeb18 | sha256-e2e0496713dc304f243c1f12476e7e8e371c8c83f84494e567679e6a62314ceb |
