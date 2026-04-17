# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8731aca670fb1adc2a11de661b208e90de02229e43a59b819be0c26634995543`
- fixture hash: `sha256-b091c6d75f126cd4fa41e0e62e2c1bde2a5cadf897b977dd808714e16a9eb7f9`
- score hash: `sha256-8026de48e70e53fc3fa9c6db76f7717772fa3f0cc957a36661c576aa0ac004b5`
- bundle hash: `sha256-32ef774aa01e381f21c86241ec46bb0e5be782ffc196b78a57f5b878bececaa4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dd25e120884595a4500dd8027a1e5e49f93c256e2e2739aa127521c9309576c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a0e5286bc578f6cb3797077b37f87ad5d5291467c99646e123e53b2e5ead4fe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-644a355f15e222361a278782613dbefb002de70ef6d66e2b4edd206ee504f86c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e6214e32e6120370484b85f766dfb113f341208fcbabd126d51162e2c801ef8e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cb3af14f | sha256-f6f4576cf7b7d6744a61d6cb9d2d15ac1947672249894e7f7ac9bb133924b09f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cb3af14f | sha256-48710f69c24908c3485903b5c22ede5eef605f57fcd3157da997c81a39e929cf |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-13610562 | sha256-671d9e289f579405095f0cf5b9656a0cec26424e873b4f68add09cfc84ef360f |
