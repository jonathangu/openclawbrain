# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-41fe4ec7878578e538ea87217d0f1ff26ff7c1c5009495fddc6dab6258a2bbbb`
- fixture hash: `sha256-49ec3fe73495575ef4a5edbb2b2c58d86b67a86df5a7ca6830045265a7717b0f`
- score hash: `sha256-77947aee085f21e65a1ef141ab881667453b89d0e0341dbce460c491b37cd82c`
- bundle hash: `sha256-972d6b014a6807a65c147a9258c521ccc6da268a826196935534fd12e2b9164f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5a96d1404eefca9da775b3bd1f6864e8e794d06c6c90d16aa5e90455db3aa7d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-91c396b70a1eefeeff54d9141bbd2cc4b6a12e26aaac0c3929ffd2e5242152bd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-762a16ed04555965e4ab0e37a680e61e2a4dc7928719b06d63149edb97b692cf |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-08e157ac1be673bd466f54651ecd9dfdbb6a032df6e0a981a6337d1d8d41ae19 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8410f978 | sha256-e463c9c805228df6434005b0f307c7647dc23f8ff99e988ca870e2bf13b62bf3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8410f978 | sha256-9431ec41bd9c52abb3785baaccc6f7f833426ae8cc06898e7b1f1aca4520ba4d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c406548d | sha256-5a9b3bf30909868819d879901169ebf0d654800f965db8195d3caa24531aafc8 |
