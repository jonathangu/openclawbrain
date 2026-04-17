# Recorded Session Replay Proof Bundle

- trace id: `live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1b2b13ce0910158e65496491abf6c903c5dfb4a0455709d2493e613749674539`
- fixture hash: `sha256-f054489b05d16d5a9f9a5c47426c143ae1eefae16f2ef4a677bba49745e4b5ab`
- score hash: `sha256-54ff3f690f04ce46634ba6e8a9d28dfca9ec7a5cc9d8f52f0b27b36061773dca`
- bundle hash: `sha256-b6e12a490d881ee907a56ca0bef7e5e3d1e93a6c87a13c1772cf8329b8962e10`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b85312fe12721c0ca336fefffefaf7611e66d0c3fa24585f0a8f1c80b737da2b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13d28561a3c9c287aef07c698159a9e9384cda134a3d60cea9d3916a39b778bc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e9351cd6aff9f559ea708b517cb6eeb184df56d8510a1c97c67b22d65f52e0a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2596ce2f7047b7646df9b476ddf07b2124a8c4a0cc33bac12b0768541f5110c5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d88200f6 | sha256-650f608aa5fb2358dac7cb263f657612ae31d13cd782c26cb65f436a08b1fe78 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d88200f6 | sha256-326043936d34f55ec004e591d053aa874de7f85fc408e5828b46f2bf89a7a906 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9133ed83 | sha256-b3847712ee53ec69718e9d119ac674072ac901a5af01bc986d6fcdd6b529ae53 |
