# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8588f62e6cb39b6bbebdb00e938513a4cbaa506b41be87532a11b4304976dd66`
- fixture hash: `sha256-523f979d52f465f7796de01a235f1b7bbea1b624b0a2f4aa71ab4b02e1ae0958`
- score hash: `sha256-77cd2ff3417b5b39fca26587c4a713b1b8e167bca8d8f5040ffdb6e0ba57d77c`
- bundle hash: `sha256-800cc8130a6343c1de32d2b29bc21bda40828e2b0324055fe62a03d8ad2c0f7a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-172463d69f5ae184c08f379b77a680b592819857917ad8f3596af66f22037f0d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-516f570f73e2021d0eb85ad7e1dc00176613ac42986213cc89f88c6ab8930fae |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff796d7778ee858710044f0280772c815758201346de4fe29e3c326daea8b7e8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f9d35801a509179f6a21cc6045b8cd1c12d928135be9b774f59d7faf5e7dfb81 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6b99e6f4 | sha256-2f6f0bec1d194171ca0b2d636425a6a0eecb4e024496ef54936122107adb0a3a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6b99e6f4 | sha256-e27f49bfca8d350f275e7f67548033866a8601247067b51b54e03476b687cdf1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8bcec383 | sha256-fe4f383d3e61100d064f9ebabb963d5096b0135320ebe69c726b536673b0f4d4 |
