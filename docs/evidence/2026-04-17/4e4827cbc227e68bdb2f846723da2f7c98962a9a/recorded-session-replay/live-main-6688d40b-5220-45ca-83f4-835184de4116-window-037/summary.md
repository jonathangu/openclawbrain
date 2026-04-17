# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67dcb8532f54fc5f6268aaf2cd959dca249a5c5832d88990a647435a45026ce8`
- fixture hash: `sha256-53a4c9afb3f28aa87aa1b17aad9db78e9f58b7b80cd2cde3904a19a0bb713c36`
- score hash: `sha256-7383d47c718d9968bdf846766c728bda87949e7e3f50ad5d279cd5d12aa86c8c`
- bundle hash: `sha256-88d5fc805887ea19720fd41f121b3b13ab894cd1d114168e29e7668ccfceec35`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e721a4b8ae2bb9ec3999c909e1329c35bf2b76bcb692645b1624780e9c7c3c31 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-617f26f61b54783c26fc9cbeee4fd74f6e564154f3bad7807ccc8f455bb6655a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-875407862fca11b2cca39b4d3620477f9cf52accc9e5c14bb0fa11c0ea9b9472 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b69600d6ed8500ba9884dd5e074acb2b6e3824f7112050394431ded13dfde1d8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9cf2793a | sha256-af6471055f65b415994d761448c4f73414b931d974674a9ef4121fc8f7496232 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9cf2793a | sha256-c77180faf601863f602ea94f7d93c8bf92431dd78a0b6a1986b498158a1c4735 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7864fa45 | sha256-2e89cb2c059f2ab2bfa58bba10240b111c702bc8728e3dad2c2e2e214bd24286 |
