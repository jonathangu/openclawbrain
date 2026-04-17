# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-145181a4c88521ffe27e5e625ec9bfb16be2cb3d2184f357c331627e92abf897`
- fixture hash: `sha256-bcc383a8935099d2c1130fc4c95751549995a1b863da5b373b03632cf18f4269`
- score hash: `sha256-32419c937dd9aebf7151ffbb648dd40b53144521ed1c2accee3f3ef2113ceea7`
- bundle hash: `sha256-1c5f548c8846a12014415d0263d7553c70007c9a9f5b1d13499041215d1433fe`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4afc95d7380c998fcb03d6870fdcb5302e0b5153c1f87770cfff10ca04ee8cf9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c0c1aa6eee30238976bb7dd1dddbe1711b05681983e0b5ed53100c69969fb8b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0f52b4b82cc73f0c25f8f6621102a1cb1930d882d9b602bfef3263df7d8ebcbc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-55942bb144ec1391ce98c3ae1d4f1c41c984d576b3f7b2d5904a1b5df5a67164 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26c49bab | sha256-0325dbf6d1b71d4a265a0ff7a776ce5fd7fe7d072110b3e54cf0a7d61c5c3b98 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26c49bab | sha256-f17c3c42ce0928d064f7b328459096b2c407f6028c72dc663dcc25ab5c8d5341 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1327cecc | sha256-ca97b95a875bf7d62f55388120835ea6f1c76c83ea786ad669785dd26c173b7e |
