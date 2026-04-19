# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a2590462dc987ced28ec91e593a00f4b408387f6ec40a92d626a6087fcbd75f`
- fixture hash: `sha256-aace8a3fe4087409ebd528569ab1ac34f47ecd7317117709f7ec2907eaa6127c`
- score hash: `sha256-24650187595a1c6869f70944eb92b5326e6f1c963cf069041c55be218bbad0fe`
- bundle hash: `sha256-c0464d6154f03d0b83707288c02c4adbb8917a6c2d44e9e345ad806e7b71c7cb`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2d2d9dd5ce486e4334796b2692780e0b5a1aabacd13eeb32d1dca3c57b5e799 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3bb9368f09fa3dbe1441c8a98ff8dda5fd651621476bc4b1f9a2bd3c51b2da6e |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-051a39f80c4c6c62044f274c3dc1ba50fdb4d6ce3fd2a48184970a3d16445936 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e2342a5d8d6fdf6985c01467d65da145edfadf0d2fa3164a92dc415a47343f64 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3cc5e175 | sha256-89d9f2b5ef73eb0ad277c973e96f2d352593758cef5646e9bd3f35aa3078bb43 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3cc5e175 | sha256-615b64757ede06d0055bd6de007a040d3322ef954357166a9a8fd68b83f3f27f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3cc5e175 | sha256-be81f5ec817729b77641e50b05c5fff7e5774a83d0b0790f3e8c75337ce67b02 |
