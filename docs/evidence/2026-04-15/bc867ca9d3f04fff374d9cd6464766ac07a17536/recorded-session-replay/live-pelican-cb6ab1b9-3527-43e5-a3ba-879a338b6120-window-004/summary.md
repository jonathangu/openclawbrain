# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f94a31a75f6f674deb4ed72bb4e73c45b90c17561480d33d8d146b93540cfdaf`
- fixture hash: `sha256-55ad28e1e1c0e357b90d71c5a61455a338c2e0a4ef3a7f6c092d3616039ed272`
- score hash: `sha256-d8f6e6afbefdc42515781cb3b43246e899ac7b45fbb33f41fe10d8ebc4af79f3`
- bundle hash: `sha256-b19ef204e6b7ca328bd9a1cbe93f0a6e27a51cb8bfd5cced28b03cdfe0fdd24d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a7e02cf5e88271092f868ed8daefe51bb787b99a9e0166c0444d9f0e9eabb76 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1738773fc164700d50f9c8e6b2ce36c8fc6c8dff2168743f11b072fc52d09fce |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d1ab65f42cb362227069571c4eff03a4ac2308f9d4e85e441be7f312ca0089bf |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5a6d3e41e56073efed16b71ea659ba6c4e3e83753465e3a3b8e807b00fa71271 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1b7807fd | sha256-c4c1b9cf189ffbd3249100319fc4d278cb862e8081951abd4c989403ce8fe614 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1b7807fd | sha256-ef8602acb586035cc914a1e93cdc082e93d7918a9dbb0fb690a4ddf7047f6ae1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1b7807fd | sha256-c4c1b9cf189ffbd3249100319fc4d278cb862e8081951abd4c989403ce8fe614 |
