# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d380c2fd8773059a5893ffff2d380e86ca0f972a4140732a5832ea7865e5c2a`
- fixture hash: `sha256-55f386f545922fe7856a581e64b7fa651b1de1ce7956a55af05d3b2bdc86946b`
- score hash: `sha256-780681d3bfaa09dd29b5c82f4bc9ec586c681483da32c05ea5c9e30db8e288da`
- bundle hash: `sha256-c80e1b2d0a0bac3674588799262d790958de68d2be88b9ba6e957602f6ba254d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63284d4d64db3291399ac8e17a28a524a22240af2c68e8497ef443766b42c4ca |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f255b2a78cc9a247c0f53f8ba2bcae16080fa8b3403daf5dae66e211be78daf1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7c8059cbaa4f7749bd18c0951697fa3884350d6e68f6a5adb460ad1ef096c6c3 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1ea632d2905b0f17865c5d35ddb26696ee195e90c66c4fb6e1d43326a180e2df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-386fc6ac | sha256-7f5703790723d1d8f121d8e89ec8ca1c2fc74afb125ab0783fb22f5a083a1260 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-386fc6ac | sha256-486f6931eebc94638ba93fe8933a3ef62f48b7fdb204c03880a1848745048697 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-54832127 | sha256-95f29395d73f0f94d045f2b50bda37f946b2d5a87e38b82f04dd721d82435ad8 |
