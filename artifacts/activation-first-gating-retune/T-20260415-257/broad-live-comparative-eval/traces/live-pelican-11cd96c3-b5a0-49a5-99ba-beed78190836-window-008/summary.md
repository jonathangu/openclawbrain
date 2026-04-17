# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fdd527bd79d12179b9a91214346f01f93616aaa30cfc7eab53977a331a071be6`
- fixture hash: `sha256-0aa39e409846ff84cb75f09fd340ba40a4ae31d0d07442053eabe16d211a0cbc`
- score hash: `sha256-acc82388d4476bea09028484d2422c675fcf2d1bb53297d675e0268e966c1011`
- bundle hash: `sha256-b11ba250ada99b7647d7a8b06957a205682ac10d074b289f362a35ab3c11d55f`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-349b3d6c28f24da121efce8d6fd84ec2564b6e3556e1440bc8512b8e1750cb4a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9bc9407d116eca3ed0661edd76185b18193cc46b3563ace667a622ad68503c54 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9d95d8fff6e523bf6e066c3e1188e240ee10c29a0e426e4d8a1ae03cb6b19108 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-2b16f61aac761ed4a7eb18a0585ccc60e96da9c46c704a6ae0b64a5f735764d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b6129078 | sha256-9530b131ba4f2c3a156011147c624d8276a1f49e0d6d937d90ed91d3b63cc3b2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b6129078 | sha256-105fd40ced0ae7243266d86b50789665d0e1b90ca52efecc3507afdf308a7fb5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-087a9643 | sha256-84edac4724c29605cdb19f4b94bbd695cc9e5ba315669eddb51e1638a3a89f03 |
