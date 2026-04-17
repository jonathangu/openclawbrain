# Recorded Session Replay Proof Bundle

- trace id: `live-main-569c731f-9a33-47a8-83f9-12284306e1fd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d27db54f25bb1682fcfa202523b5f1c6efccc7e2753d8e02e54ba11f6e3abbc5`
- fixture hash: `sha256-0e0db1f3540c6bbafcaa45e48b36b0aa0cc986ef0dddf4d7e13951d4b175679f`
- score hash: `sha256-6c6e7aeac77a7664f2863bc6bc0271ab79c97d3b161752c3816c5102408fddee`
- bundle hash: `sha256-5b004ae7cb41f8e7444ecec6a51b5722a431f2a4d657fe9a71901f6b73cee607`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-71d12c78bbd92c17749c2ba921bc24d7594735564898b2d4c08d5a5f8badb93b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-48e25329920879265b8381ad7ff10bf8190a20e8bd6c9f81c0a5f910d6220626 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54d439cf121f570a48c3f57d15c8ba1bffaa734d187b3509441367209b135733 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-67e77a202139107847f83e590d7c8ab836005025ea7fa3d9dc36848ade9ed33c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-671c953c | sha256-5b991d8ec70724054ebd20cfc292e450ef5f0c6ebe4ed907a2a1fb15e91142d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-671c953c | sha256-554a8621686e8c01642ed63cf8b3e8e98da575267dc9281394eab7861d61980f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a6a6b52b | sha256-4ee1f80d155f50f941ddb82ce7021dc38e7e8b427cd336a4cc0c2e72f4456f5c |
