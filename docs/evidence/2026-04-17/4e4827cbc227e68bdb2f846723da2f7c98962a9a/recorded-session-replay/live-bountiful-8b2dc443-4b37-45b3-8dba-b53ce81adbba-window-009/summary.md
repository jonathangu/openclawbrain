# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8eced4262f5a642239299c7c899085a7bd53ad7880d03357a10326803fe33aa8`
- fixture hash: `sha256-5aa5748a68c006cb4152d6b9766d43523c43872689382d99e9608f0fedb263a8`
- score hash: `sha256-d8c3387afa8729945c9c5c9053bc769f97d5c065a0c2191575a156aa492b2389`
- bundle hash: `sha256-eecb5f2c9726ecd0959f95cf5a9894b0968258207a0168f4983f2994ea0d67de`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86ead80920d9422dc3144931f0210740c8474d5a0351518c55316e7dfbfbffe7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a7fec9282c12cd3fc4a23154c3f80b24920e251940f7bdb4ce248b357520c9ae |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b27c977b269a466015122160b223f45cb60324787b334cf5814fed769459f41 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7e1388c88682160a5bab8367362949fa9c69cd5992786cd80714bc028119dc27 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b5ba2299 | sha256-55858612a669cb834f2d54e788a7e7ffb5141187c0a2b5c42a8e287b99a67aef |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b5ba2299 | sha256-f0fb5970700fb4be5042c8212b858723044c545127729b613cd41886e6578d3f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b33b608a | sha256-97d0920d6a285e27622f09e941c69478328fc24b03151ccf42dbffc37256e53c |
