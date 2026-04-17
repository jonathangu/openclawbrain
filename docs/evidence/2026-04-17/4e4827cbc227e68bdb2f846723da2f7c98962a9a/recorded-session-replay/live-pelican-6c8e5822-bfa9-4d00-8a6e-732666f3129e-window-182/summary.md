# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-78387f2c5e613b60125abc448bbb555d66d8f870fc7df4213c6f787249189d9d`
- bundle hash: `sha256-252ac8c090f4bd6c195ba099a740e81d9534f870d337595463f6ea5d4279e831`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-fc7bb8a07867b1f4d61d6a2f1b790a369ef8e816f7c6d30b29046a1027c92e73 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-ccc3d08899296e2dde69db9c59c07282b8e5cf7d9dc664967d6a532431ad3803 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-ce30a62972e53472b424de47e44c86c6367d341587a1ee97780897a6272b6069 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-54c76fb9 | sha256-432999a915918798d9f71bf21751fa1eabff354ee9d3a66a60b025cb2b34caef |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-54c76fb9 | sha256-a9a78b56faf58862ff20edb37b65662a45a00147bf9b964427472003e7fd9cfb |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-0b4f8980 | sha256-9044043248e61b77b27be76f163e59bc22389d1b1eba00360807ba7fa0e0e6e0 |
