# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b0cf5e7b3f41bdf7d892185413608401c24e9c3ad252c16335ba4fe2f91cdd3`
- fixture hash: `sha256-ed8248c9b476e9fb2d02b9891cc8e11da35a8ba49c308ca9793fd2e0cd5daeaa`
- score hash: `sha256-f413fdc566e07b11b9093eaaf737b953a5a5b0f57108b9b71485b231abc9873c`
- bundle hash: `sha256-574435977bb4afa223bebbd6db344d784f50a48f5d087cc5df0ba68ad030d521`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-368ba7ad0e0062707beb6bc226c2cae8531ed592ec4225d05a99c6ab4df81531 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a4cb8af0b91a69fb275b2c02ba024d7b5bd780c07ceecfa0938fb3808aa4a013 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8bf7a10f702babc9114d16a66adbb4a30d66b68999c4c7ae10b43ca06696b1a1 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4e54a8beda3a0c475ffe8e447e3ff8697f6b59ab905e611ae9ae419188c81b68 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d622436 | sha256-4ff6d5cf09b90b0e14126c7c62efa67e22cda6300eb0e2626143cf6b373a4cf0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d622436 | sha256-3087bd1df1f56118fa85529d47a8840078a5ba7a61635ba3f33f32fbcf15d1d2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0b1d995b | sha256-4896bb9bc0e3a50cbf7276346374cd01355e30177d06bf057e5bbdf28f175eef |
