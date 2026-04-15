# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af0623cd896f3d36aa832764b91c449eb65a56e502af4829ad2995082aa19cee`
- fixture hash: `sha256-729b2a143706d45b443dec7a409dfdba222ee805edd97aecb9fe78e30ae910a9`
- score hash: `sha256-aa00cf238fcb53f831ff17b000dc28a8f5386961f30d41610dd428445cdad812`
- bundle hash: `sha256-99de50cd0aa914cef48f6227baf2c6491d9957080734146ef2f703058e47b023`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72dc7dab3dc434226257b098b5889b33f6d9a175c84b5a7ecf9e06dde7b7bf77 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87715157eeac1681189d4a8aa8754cd2b90c35ea5e783062e9cce2afaa55567a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cd7f51ba9e844696b89ad92437282e59e80573ff685be7c881034b0e94fd7ac9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-629f52a00a31b5fed4e0d86edb8fd8bd2a00f6135521ca125c4d056bbb3e9958 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fec6f63e | sha256-02566b882a13fba24854aebe8cc7eea607d834d15bb8f05cf5ee19401c6eb6de |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fec6f63e | sha256-11974ec6d050a56f4ce9a5196095e69fa289a767d9224facbebbcb12b56f870a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fec6f63e | sha256-02566b882a13fba24854aebe8cc7eea607d834d15bb8f05cf5ee19401c6eb6de |
