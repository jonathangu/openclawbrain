# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e44bd492f128c06a27f22e67cd820199254d2a3ed0a6ac13485df4261f57fa9b`
- fixture hash: `sha256-cffeca9e647d7d047b9dbfa0c2bd2eddc1a7b9897467d5e861f95728aa0ee6bc`
- score hash: `sha256-2dc97035d5813506e5c5df4f6965d3604434de7741839eb8bc5d251d3964c414`
- bundle hash: `sha256-d1cfe56256495542eb75a26cc82b4a0344c0491d11fc0f14313af88e5e2559a4`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2666a690d516fa04e3c9ac25ce3645d95970eeaf72ffe7e4a929bded5360f2d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ba0902d46d726507ba19c4c1e444910b761d2e8b2f47699d5b1173c55b7aec71 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5672a5ce3bb3bdb7a81c8f63b4a4f8f759e999302a38fb5e30242f4f2939833e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33e8a791 | sha256-ab06bc6517e0c2734319f9b596ada03311f2f4ef36cebca406213deb22f58828 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33e8a791 | sha256-b20849bef957cbd5d4c0cde336794e7f72b5337beea78cd516f4c8125270daf8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-21dd5258 | sha256-0b1f8c24cedd7ebd54cb69672e6a72b7f0051ad7856beb6b71aa60cbdc5801ba |
