# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37b9967646ced8e1a7e53e66d95e96c0d5cf9872e9f6cf5f223ff75c45212fe4`
- fixture hash: `sha256-e5562aca0bd9165edb9d4f0591f9dae6981c5299e9b8cff4453286d3a3e6c950`
- score hash: `sha256-339826197a6c4ee88ea06746183ecd48a0b1212b0ca3d11abb125fff81d34bf8`
- bundle hash: `sha256-89cc4e97fe4af630a208d014fa9d7fe53b626a0f194a160bfbef716276ed9075`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e59e0368806e0012160cf4b2dfced7c5e08071a2c01bb62268694e031a82feac |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cfc7069d484519a619424285f7d93878966d875fd659683de38066f9e87b77f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b541103c2bb458fb3c5eac6986e8c436859e80827f7e999a40d932cfd8f46cb |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-77d6c363344eb164e997977f117d1f80355a4b8bc7be2fa7a5477b469d78f833 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-917e6e5f | sha256-d417ba9c90d42795b95f4814624535aacbfb3350d49e93456900704c4cc874b9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-917e6e5f | sha256-ba455838b0e826de7497fcd813fad502b4c1e33b83e5f6a92b6f1db20e831161 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c25bc214 | sha256-bd5b301ca9981b69fe40111e750ec8c44f2495453ac3526106b100094634a4ff |
