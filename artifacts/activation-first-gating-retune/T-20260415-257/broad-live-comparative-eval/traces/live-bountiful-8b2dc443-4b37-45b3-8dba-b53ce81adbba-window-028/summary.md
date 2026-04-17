# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e5cecfcda3863d55354a9b67074a3c6ce69c277ae2b3137a3f72bd0dc80700f`
- fixture hash: `sha256-6958fe867e36da1beab1df863be77bc3ca8278fa4e3d5aeb7c88307e08cb7f39`
- score hash: `sha256-0a23fb661f5c67aea76b88b7542ed4b33a2fad78d031dde15c5cf3ea858665db`
- bundle hash: `sha256-9f0124fcd8bca4a457a9464f28c98473bc39b3eea97002acddd557baeb10ab00`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7381932aea4d1bd30c10ae36d19326006a8cb4cb3b6e5b2b2ae6dadf03b6d135 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-805e2285da07fa15c2f42017597c8bb8fb1093e2e2d92c568f53e2048e9175ee |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-58b178e5d8c31b8eba1bee89796e99c19e519c5a7263e647be5e984b6dae87e1 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6defdf3c57773f39179cf259d18609fe1a7e98f74a6e99bea11ef638e7ac3178 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9b72b190 | sha256-db4c4ca9fb147414e40592870a62252749b80163b4737f2bbb203243bf7ca99b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9b72b190 | sha256-4777092023a81a459e9e1bcf70569067e4629c94071b2a65eef597195b360bd5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-91525ab3 | sha256-b06a8da84ddf579715b38e07c05a65c2e45351e8b8317d56b5949741df29cab0 |
