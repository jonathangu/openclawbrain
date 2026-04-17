# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-166`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec7f0ca39d2ce8e4aa075852c984c14df45efbf7ebb099adc4d8318c646741f9`
- fixture hash: `sha256-1eeb0e4e14f003831776523471001891e5f51483edf8cd0fe82b3b2a7a4e72c2`
- score hash: `sha256-8fc8816c4868b75423d99593b6cfec35bb5697a192837f0b332aec265f057e13`
- bundle hash: `sha256-9da707e2f4f0b12a8147afbbf8ad6a673d8b607a1ac7f58f692d037d71eead9b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d39232e9e4182be91b475d1dc774e142ceab1f9213fd98395428e4f29aee341f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fbd86d2ba1c947004ec2bf7623edc0fdbff6d0857c9159e845ec15e09fb39046 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-822e844e7c10184aa9b853d2d86b3e927e1b36be38cca40f38e300ef690094fe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9d9dd53ee584eb233632904fbd7e67174f24b5e3a5d2695af0760f79d4c86abd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-30cd70aa | sha256-891a8c530bf2bc23e71135e408bff5f31cee4fae0fd0c5719de7fcf1cb29aa37 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-30cd70aa | sha256-a3ea742ef3abc2c91dd00d2a9e0b562637a63c8e093b723038582b533a5e99bd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d3c85a37 | sha256-6cabd887533318537e22b9fe35fc265e926789770c79b0030f17f618297ac14b |
