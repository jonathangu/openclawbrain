# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fd4a73ef0679d3bd5e8a41ecf8528eaf1056f459a2933d6bce7a274e1da6704d`
- fixture hash: `sha256-cdbe046df5ba47eb867d34f32f856111ce7f2bac423e41168b29efa3bc680b6e`
- score hash: `sha256-941d952963f1effe4b3b87314bd0c78b78455b6a80107e42d8867b2dcd44e16e`
- bundle hash: `sha256-1ad0bea15c532688374f2e5b49a8bcfa12af9f1d8f2b011f197646a950ea3e89`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-195c8562b43d566f299d3b4d568af19c059fadcd5ad0dc52c1779f850a2eeca5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5443fab869939f8fd02072c0866be732f9c21ac64e3972071a201b31bb767136 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3191d5b03846a3dc46f1a9ab1a3c92548607eefd7f05ebad21085531731fb485 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-35880ed80b73ca16c74bf51e6f14880e1d1684aab6d2529f36b49a559ee552b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4015cfb | sha256-1eff69abf0703b5c642a8027f0434b65a2f0e20ed02bd1da5142bb800b4566ea |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4015cfb | sha256-0ee1bd902625aa910a504c1cb17e2a1285df25c2cf0dee5bfeccfb50ce5d9fcc |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e09322ce | sha256-767d7cd6093dbf5f118e66513c8b92f5723c9f515a83d52580f30192d857b198 |
