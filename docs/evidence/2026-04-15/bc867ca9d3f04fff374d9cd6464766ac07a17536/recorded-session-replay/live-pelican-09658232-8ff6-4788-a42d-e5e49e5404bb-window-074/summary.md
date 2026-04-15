# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-145181a4c88521ffe27e5e625ec9bfb16be2cb3d2184f357c331627e92abf897`
- fixture hash: `sha256-bcc383a8935099d2c1130fc4c95751549995a1b863da5b373b03632cf18f4269`
- score hash: `sha256-58ec2f438c72f2269e8172dcc5b1b9e1e943d6382c060f9eb3f975312be609a3`
- bundle hash: `sha256-b31ad3f0da26d4684db21bf2c616608b19d40856d0349135356f71cb3c0f762e`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a5c5a072cb06e2a0ccc7825f51a652d340067b140a19a48e400d6a98f2ca326 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e9f8d2bbd84907eefab9c3ea23631b1d4b122601efcefc9687df65c20ccc1c99 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c9652e538dbda5398b3bb31e4994a53266b8ae1eac652fb2228cc5bd00803f3d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b933fe5e | sha256-41a481fc0fe24a7dc9e019f55417f34aec85d77d777e80ecccca720697863858 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b933fe5e | sha256-cca2387793ba7d68c64ebe75b89e32a14dc8fcb2fd305948bb0ccab227a7e7f1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b933fe5e | sha256-41a481fc0fe24a7dc9e019f55417f34aec85d77d777e80ecccca720697863858 |
