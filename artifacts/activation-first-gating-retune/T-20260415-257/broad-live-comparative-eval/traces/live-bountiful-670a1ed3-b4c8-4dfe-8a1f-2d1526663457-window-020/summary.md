# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b202c1c438845d3c1c73ddb7c1ff7926a10fda7c3a64127ae541d469c9475d5`
- fixture hash: `sha256-b48968b0fefff768efffea4ced309b4343ca39a6dbbeda150f150e0d012ef675`
- score hash: `sha256-bfcb51fd825ddbe2272d3185408dba5e46890a318a1c194b044329010ea2d2f5`
- bundle hash: `sha256-4cdaa58ed82a13c8c7799376ab1bd5c251bc0ade831cffd01952c6ce284f3164`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-07841a59820286934b7db3a291f9a2a056f9291d9bd4bd106e744c3a6ac3c6f8 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-984d4aea3d0971bb27ed782f999e7d14c01c01fe6d3b7bc06bc27259a9b17fb9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-69f9a3180f8eaf92e39841ea0c50b165023f64b66366ccd521862065aa272438 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-61a0c2bdb5b69c3cecf8d503fc80c32c905d61722b17ff4c24ce51bb1476f688 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f8de8a7d | sha256-8890a004ca424d3c318eb849c8cf479dc2f884aadd7f9be4f231a240f3945433 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f8de8a7d | sha256-e41e8f172473f6efe987719f28677280034859536783ed9f7ddea8823cd2e9a3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f80ee670 | sha256-31bb399e970b550f7710c54cb81d33f875c5d19522859dada430faa983a96a45 |
