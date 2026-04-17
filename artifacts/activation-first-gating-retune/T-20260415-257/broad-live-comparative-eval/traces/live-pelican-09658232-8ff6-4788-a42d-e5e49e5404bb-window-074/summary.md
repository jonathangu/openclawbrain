# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-145181a4c88521ffe27e5e625ec9bfb16be2cb3d2184f357c331627e92abf897`
- fixture hash: `sha256-bcc383a8935099d2c1130fc4c95751549995a1b863da5b373b03632cf18f4269`
- score hash: `sha256-d7af1d8f48a5c399a49972170fe4c784bc7f94922995de28cc34086e0dfe64b5`
- bundle hash: `sha256-1ca0117101ff21485437a3c789367c99265d419fb4352c3d86e24cd34fa0e083`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4afc95d7380c998fcb03d6870fdcb5302e0b5153c1f87770cfff10ca04ee8cf9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f99b145ad962474fe48cc642a51294dc58512989a234b425e0f791eb2f0948b5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16fce90a48ca5176ac78241ba8eb7237b7c6e428636bb586beddd15224528dd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ff176eec07ab8200b12d5d4561735d87738c4c2dfc2d228c6acf123f9a1fcf77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f0865969 | sha256-7e4802f74f3f354d209276bdfdb6355504e1a6734b278932ba3ba4d4da6052b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f0865969 | sha256-4df6c6f2080390eae05b3758f7985c2ca1934f24546b7e376a95ec55df0b8b25 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-dce98c8a | sha256-185b95a3e70ec61a281bed64968646bad0d16c51b1f8a3a40797ddb43525e262 |
