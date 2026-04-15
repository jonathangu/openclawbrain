# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67dcb8532f54fc5f6268aaf2cd959dca249a5c5832d88990a647435a45026ce8`
- fixture hash: `sha256-53a4c9afb3f28aa87aa1b17aad9db78e9f58b7b80cd2cde3904a19a0bb713c36`
- score hash: `sha256-404d9ccb3a5259bdb095c8efcdd22284673ace63a1bddeafa5f7473782f54509`
- bundle hash: `sha256-80a1e4c747ba5019cb5b7b4e975306d6d16fd707b2b73e315aec0634c2189edc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e721a4b8ae2bb9ec3999c909e1329c35bf2b76bcb692645b1624780e9c7c3c31 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-21211c0b4482e75434a40db59aea315d3fd3b809e4129bdd4ba66f9636edb198 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-87a33b907f2017661fa715e68af8fe706888aeea97f13c37827a68921682d980 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6e7b7e71f498f5ed5b7ba37bcc457b0bafe912eeec7e105a6ae841283a2b27c2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6798563f | sha256-8c4e1c02c7785190600df97d0ea783a67ae5351a812b9519141933724cdb8d61 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6798563f | sha256-246282bc07597ba51ccad5fd04f23885cc8b61f5e7a994a9a1353f95f9869505 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6798563f | sha256-8c4e1c02c7785190600df97d0ea783a67ae5351a812b9519141933724cdb8d61 |
