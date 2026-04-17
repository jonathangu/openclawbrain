# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-069`
- winner mode: `graph_prior_only`
- trace hash: `sha256-836584be6983eaf5fa6eb8781cac34bfbc9fd538fa4b161ac2d1263fee14146c`
- fixture hash: `sha256-d2752f3a765e793797ebfa0ab38ae1044dcc8b2c28b548d73dcfade2be50b251`
- score hash: `sha256-3ef73429b40212da435ee0a3001c0b3d8b48542a01fecc5febf0c28dc9f66bdf`
- bundle hash: `sha256-6e4aaeacafd957854104edfbad0441a2a00861505b60a92c425614a18cfaaef9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c391bbc2363181b72ba8549d9009dc5fb197cd45a7341e37f0fa91e51803c6d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-65504403da0d239d18c5c0e567e7da63375f8d5763ec1f79d64d47aab42f305c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a01bf96bcc57fab9cf2f8fe586e29e464461a4b3907d7a9023a7a6fcb9e470b2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7d070d07782b6641ceeedd689c8cf5016b6ac246f5f458f9231bfa8b601c5f24 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9f969700 | sha256-df1e5723f8c9ca105ea4655e71e245470443863e475359f121e7bcd028697b98 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9f969700 | sha256-703fce1e35d8ce1c0076cf92399600adf184ecfae8be38046342a6ae3789ea2b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ea31c8a1 | sha256-a92715252e6c99baf8efeca1fa60594707aa8a4377ce4602af16c7613f18bb6f |
