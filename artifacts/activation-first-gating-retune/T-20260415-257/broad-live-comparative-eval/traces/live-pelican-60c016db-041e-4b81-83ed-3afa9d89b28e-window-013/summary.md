# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e2a057c58ceb7779d689dcd4238dfbc3207e352fc341de03ac7a06d504301da`
- fixture hash: `sha256-737f6561e785d3bc05d3981f983d5cf16785ca63d2f46199fbc1baaeee1f2b69`
- score hash: `sha256-04ca7aebd9b4fb2117794feccac60796a45e7712f701d46148db40d37256c2b8`
- bundle hash: `sha256-509b60ca0383424204ce1ac5aacf49cb2ceb2618920fb6f6fea373b6a0a96f13`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e7a79c157dc055e3ad83a213c22e42badb5ac82b3ed30aa50ada887959b805f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bdaa96eca600d4c11a73b6fe9df8ec48da29b6d703b864c84ec803aa55c78837 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f1615ac8cd9c909a2f721d4ae71144e931bb796296a72c8bfd1c2b3508257ce1 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8c6b072cb4a619b5c09fb110f5b74447b688d95bf2632adf22a959ccf9fbf570 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d5da88c6 | sha256-b1f70124f45b83e78d32f4d0de7ef12d63202a3d7788857b9a69332897989337 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d5da88c6 | sha256-b9e502a389eef8673f620ee98e6bdf501563393c0c9036e50a251f424592de18 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f748b899 | sha256-17abc365a53b43fc8037192ff724352357a7969d983547002e7787c16a2dca83 |
