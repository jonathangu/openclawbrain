# Recorded Session Replay Proof Bundle

- trace id: `live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d82a353da8dcdcb26be266a0d33a583d117e1f2075d582930b7ead32e3d715fb`
- fixture hash: `sha256-3e1329cac030395635745dbbbafec0f460454aaccbb63b26f155aed0ae65e7c6`
- score hash: `sha256-2f8412ddcd7f31f48b704730228fdde1a7516bdd2cfd16e35a72942ac3c217a1`
- bundle hash: `sha256-700553189ab807be324428c441ee393e00144902953221cd642b1510feca6b86`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38cc6030cda1cec211011cfcfcb3fe3c0763917e1cb1cee36ef6155b409ff4f0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7bcdce2d1eb35d6dfed14b5ade265d12e25db2bf530e6e0b8d770d0d1bca400b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1cf13744d7efe8307a65a95fa82e81100ea93df33158b7452ee4c77184ffdb57 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-b984b36e562b6179d39f809c2dcd2fef836d361c1d151d82490935291103ed8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0cb1a6ae | sha256-db61d04c8e5f440758be542126fd2815a301d624e5379faa7a105116bccd4844 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0cb1a6ae | sha256-282ef323cceee9f3d390873cdf3675300c15b67f72bc5cb27e2a9d4a1711f692 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-b9e05b59 | sha256-9fe4f86a2f168d9d0a5a61619da8bda34a335dcb97a2820772fd2d63c58678ed |
