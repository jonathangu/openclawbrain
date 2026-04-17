# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e44bd492f128c06a27f22e67cd820199254d2a3ed0a6ac13485df4261f57fa9b`
- fixture hash: `sha256-cffeca9e647d7d047b9dbfa0c2bd2eddc1a7b9897467d5e861f95728aa0ee6bc`
- score hash: `sha256-96c42b44114ec7b60b42a13bedf1cdda4a8a6eedc81bc6e2252409bc71cef28b`
- bundle hash: `sha256-07a1a46509c794368b5bce9b2d6730c0b33b893ac7de1c9c259b43725afb6dad`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7211fa1ebf40e19b79ecf69c6d2f4cdaac759ca9e3451e680c32982ba6c5891c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752b4d9feec15a55c5c1cd665b444894364efb2da12b102cfdf441f217e3f54 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-825f1d713392bbbffa204f3b61d05dff5c9f52119778d4191d015bb62a556e7d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-73e47eb3c7506b48cedc0d10e0e24500c3f5ebb398b173bff9fcd325803e8711 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-85fcc6da | sha256-7603344a40b4f6fa5e29450859d476467f1a14c19487c5a6947d5eae633666a0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-85fcc6da | sha256-a20625e91f991e5ac34a0c64dfc2328c3cb411253b10c2ff4d97ca2b3121467b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-73f171a1 | sha256-1dab3c91c29173e172cc405eaf1937e049c97f518655176a3a276d749a854125 |
