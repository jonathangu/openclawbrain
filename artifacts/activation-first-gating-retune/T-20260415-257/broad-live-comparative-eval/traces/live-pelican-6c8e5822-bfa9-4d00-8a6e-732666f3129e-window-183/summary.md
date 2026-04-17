# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-183`
- winner mode: `graph_prior_only`
- trace hash: `sha256-203ac39480367005fd42cf7825311a0cbe85dd80f56721c12d00a8ea3f270b1f`
- fixture hash: `sha256-f1b7e7068a4652fbad5d085cdb0c1a635468b0ae89cc507258e65b4da9413c08`
- score hash: `sha256-17867af2591225765474ac3ad83d6ee125e6cc4aa6c126e9620e6cdb424494c6`
- bundle hash: `sha256-b9ad3efe60c9d81ff5570d44e8c1a57c912f387d05af137341fd255ad588ae6c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-08044093faa209a549cf6cbe79d77a3fd872d3cdde2c86b5886da5044f650477 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6a8569fb411ea5a982f3d6e8da55e9b052c55c4d01e6a2acd87f66d854de241e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-02da0fe6f78605558965bb8eda215eb450195000d903b59df6a20c7e2a792b60 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-7738ee3ff39d8761f5bb6c053e96c2655564d885d2ec26cd3100397b3dd35a4b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6a8bb53c | sha256-8d8723b1c5a53975f9a7f7f18de34609c34dfd54e9cc2e89efaf97405db82e4c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6a8bb53c | sha256-0713c537d4ef2ea66be380f47c342c0f5140c3be92f2167bb760d25fd2ba74fb |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d6747525 | sha256-df37207286a58220a2e5db87e9476ae81c86459572816b36f1d71f3c38f94bc0 |
