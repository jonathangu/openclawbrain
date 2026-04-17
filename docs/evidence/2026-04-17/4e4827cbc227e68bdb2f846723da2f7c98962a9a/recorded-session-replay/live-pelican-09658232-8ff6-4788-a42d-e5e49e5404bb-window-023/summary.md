# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f60100eb1742bfc6c299af2f5afe9b6c211473ff986af1ecb211b198ec2ac6e1`
- fixture hash: `sha256-7060077aa89ea2d2ed121c14a4166c1764801c149a1d2df1467761d22c2169ae`
- score hash: `sha256-7b18d5ed744ec791be9cf0b5dffd0392ca62d6bd429ebe2f96f71c132f309418`
- bundle hash: `sha256-c843b2dcf5e3db98606a7694721fb933e9102a837adc24a238fff5b619d49e23`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1af67f3f0f2a5d2c63ece4b570453604d2bc85441d7219830f849b19b9d0d604 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2acc7cccd7237beec266f2a64abbc1e853219b05316b3e1cb7d449d3847411f6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10e2e78f22aa178cafe93b488e6c008d91ac952f0d4a5277313efc004c380dd6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-48e29312cf36b5cca35d2eac410bdea3c29a7435440ad65dfce83e09fdee9c6f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e0bd4750 | sha256-10c3d79a3c82033651f3b54d2e4c0939468e1602e23dbbf5630b45c629bfb761 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e0bd4750 | sha256-e0ca1a1d955ccd8cfa49cad429b6022af1aebf26bd29170c025354b270052f0a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-639c77f7 | sha256-f15a2dc65719a87d1999746f6796634ebb8f382c00d2ec39b0a2b9c67eef1f55 |
