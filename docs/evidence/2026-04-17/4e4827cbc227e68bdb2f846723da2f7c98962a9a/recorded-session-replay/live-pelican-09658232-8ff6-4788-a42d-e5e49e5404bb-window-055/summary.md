# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055`
- winner mode: `graph_prior_only`
- trace hash: `sha256-069c659e483d79099c9522902169a0e3c008a2a3a1a608f281e5842abe60c793`
- fixture hash: `sha256-2053a334b00cb8986b08e94b050daa206b7253e27c1f42496d3a7ffe4c19e5d6`
- score hash: `sha256-a8f2221cb387bb09095916fd86c5eca79a6891ea11c35c0ed1a15bac7340a861`
- bundle hash: `sha256-7893b809df41e6dc3efac4d38a6b3b619fdaa2e09c0ca043a92fad266ef5088a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7d0a9be6adcdff721e255d979e1bead77026cd16f5da5ab306eca424cee158d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8c99310f94f82014bb42f7de67403fa8ddbc69ab639ba8c87d684153ee648be |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0eb9878f036fb544db97c62701f211e0c5ee859ae2548d5585d4b07c1768b24 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e4e067c4af37475ff919a00edef217ff4ca1c639bdeb8bb9c23a0fa423a164a3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a4d06c0 | sha256-89e72d1af18efbee7d1ac8625b1499f821d393b3982fbfa28b4f9a4b2507fd8c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a4d06c0 | sha256-4ef68a3d72b1062296df47c41fb809c7608ad91abdc9dbfc0ce04da02dbd60dd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bd5da6b3 | sha256-18818cd10aa627d404d2b7afb24c56bce7e4dcb72843a37d97c71cba1e10ffb5 |
