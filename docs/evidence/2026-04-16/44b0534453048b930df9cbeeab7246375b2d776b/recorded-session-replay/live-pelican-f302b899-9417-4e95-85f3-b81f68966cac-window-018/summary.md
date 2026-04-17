# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-367ba0e9c1765adfcb55faa49a77e3f08a37eaf77c4964ca4eb0f5d706e75deb`
- fixture hash: `sha256-c755dcbf454eec2e6cb44da638da71dca0e7b64e802782c096094c2870f2abfe`
- score hash: `sha256-2e88dd27d026190f52c03b4da8407def1fb9ca1aec8dd0c8a9582cc68b97da12`
- bundle hash: `sha256-a75d65cb76c71bf44c4fbe589e3d1226fe8f26584954704b3f59937ecb0a761d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df30ae86da7dcb946f187b86df35238c1caa6176c275bd81d1099e4de3972842 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9be11b6b0bf946507c50514f386c9c9ffe8dcea8ddffdc7d5029c77ce39ff7ec |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-163a6e0b3c99c9566f0535d1b1ddd53bd4da5fe247fc21b5608b99e6416dd59d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-438075e6a207c773979b49c5fa152850d5d28db9481af6093aa675a3310e1477 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-df27293d | sha256-ac5fbea1ed93bb99888ffa6123a41aba979d9c05e7928619bc48619a3d008c2d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-df27293d | sha256-ed37f0635b19f8fda41dff2727dfad5588a7064c4f46ac6155b19cb4cfdc6500 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fb0cc486 | sha256-8b3a426f26df325e9320b56beecbe9019eb9a7668ef9812c5f1795af2224f94c |
