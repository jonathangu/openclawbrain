# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84b9a4843680de911479c2420a8592984c3d84b3d54d06debdc96d5c918ea030`
- fixture hash: `sha256-be373dad3e692162d5000f12580f9371232c68a9b0f09d3136130b3fe2a640e9`
- score hash: `sha256-764f94f6f065f3532ed928122473e5c73845b2fcea0e60ed690a82130ee12314`
- bundle hash: `sha256-3f390edb04cd2b592629a7ab27f881d50985a2efff1f94bfa76a809589a16dd5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-34864147a65f338d5fe87baff27e70ea8462feed84ac2fbd4644ab5e3e006364 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-946e042c40c4db75223000151b56f371be42e455f3745eb805d645ba0b017e21 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1181ef28d777b09db259f9a8f6ad4784ecd8b20567aa7ebe2a941d8ff9f3a189 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2a5ca719f220901cdcc45b31ee287e624a192cd88064bee39cdf4d8a6a4d19de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5f4b484b | sha256-11f6e521c393652dd5d49be94d8eb29586ba77886908f119d07156773491db54 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5f4b484b | sha256-9963b42ff08226b49cc12406fe444360200174536951bc58ca0d40aa9018ff4f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-12a1768e | sha256-a2eacf26f2025e02b4d0ad2b7a35ce97526e860adcad87b8959d5af6513feebf |
