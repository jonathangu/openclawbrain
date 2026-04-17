# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0107560e9fd434b7938c996a94e09516e9330df1381928365035d337054775c9`
- fixture hash: `sha256-ccf24038ed94c209310a49ea52fc2105449214d461e66d2dc1493bec54050346`
- score hash: `sha256-4bf872dcb4d7f0f3fff2312ca80dc2dbcef3d6e33b035d908afecc1e1f15c703`
- bundle hash: `sha256-a7e30a26c687996bea0934e0714493fc442d48b89dc2f32536942d57a3ebaea6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cd39ad77bdbc78ea0e62a163e0f69b70fc53f35c07ff18076ffb99dd86c22c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-afa7349574f7569f32fa62ed50123bd7e02fb8ca4e8079bd1c15fcf9722d0fe3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eae9ffc908b6d0f30eaa73fe32a9317deea10298a8443c8d34c0395057f2e909 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-028f72fd5a4fa5a5f9b0bdbb9bedd9e96aa40b143368302ff55e5327863da281 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bea2d362 | sha256-9fdef2047ac1aa0c227163d6878367d1088b6e8ae6ba8fc802cbebe03a328207 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bea2d362 | sha256-a28d8f941e8528e299d9c581ab8a0b612d7874d09508ddccd197b83661d39123 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ff6100f5 | sha256-23e1ccb9f52910de52c7d7bd07d1ebf68175e48cc7fc8483b52a58bb94e6ec32 |
