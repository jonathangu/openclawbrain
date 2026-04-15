# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b202c1c438845d3c1c73ddb7c1ff7926a10fda7c3a64127ae541d469c9475d5`
- fixture hash: `sha256-b48968b0fefff768efffea4ced309b4343ca39a6dbbeda150f150e0d012ef675`
- score hash: `sha256-1293185f63f6c4638803b1bd7325ae88f8f6a3247597d2b179ac7a6741deb010`
- bundle hash: `sha256-04ef2fc390199acb9e0423ec10f6393f16dec1c8f62789ae5a62be15d6bffa04`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9dbe3c29767c6133223d019436a30e66b07c3c01dfc023d89a82177b91a3054f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b6686b41523c89f95587a79777fd16cfba8ec9dbf00201dce95bb0046fbce47 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-56725bff888cfacdf3fabb77b516cbbdd2c5b59b16e5bf087025bb1f041ab8d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78f91591 | sha256-83e0bea38de39d5b213faf74f653f8bff2c2255f1212321576be9aca88061844 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78f91591 | sha256-12ed98bceb5418d47b9711103dfb9dab72ee862862258c3901ae5adc6caf1c32 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-78f91591 | sha256-83e0bea38de39d5b213faf74f653f8bff2c2255f1212321576be9aca88061844 |
