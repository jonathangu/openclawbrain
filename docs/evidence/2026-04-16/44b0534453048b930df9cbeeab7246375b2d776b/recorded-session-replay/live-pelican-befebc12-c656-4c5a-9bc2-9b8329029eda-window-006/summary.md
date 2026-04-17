# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f9684c38d91e55a983d42052df21e03bec407bc3f34393946fcda8e1b2d39f4`
- fixture hash: `sha256-5b6e1bbde60f4bcca2052f19249d943d07695521da1e7e8b46846e97b143bb5b`
- score hash: `sha256-53278a81416a1bf026d756f3c019ad186954a250a4b32b890b58285af1765c56`
- bundle hash: `sha256-df728a998acc5b2116f64762a071e4f0364c68c5f6ce3fc8098803a3aa9a7373`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a0ec494cc9b5ec66bff70ca0bc3e9262d5754f8a93ce7d222367da206ee232 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e07560b17ffe971e5ffe0429fdf7f6024bb514dfdabbcf0384b4556a1c67b422 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6f5febd262b5078ff1dfd13b7ceb1aae72a145fdea9feecd9449a15bcf2748ce |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-14476243881fd52ca09412a3edb4a42761c5c311b45d1df0d8d68e184c1b9c48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0b3a8f6e | sha256-4441b668257e60914d305da3941b2bddd480625b5816b6c7fad7d5dfe30bab99 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0b3a8f6e | sha256-4c7a26065af8125349476ba9add6440415ccaf6df7e2988dfa48a039433bcec3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-4fcc8739 | sha256-b5adcba793bde8a3aa7cbac67e50f0834a7574697218da2e2416e47017c61969 |
