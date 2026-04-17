# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304b5ee53cc148670256892da800bf0d31f07b699447be9e8eaaeff5a3c2cab5`
- fixture hash: `sha256-60dc2f86ac1ee754f931ba95c5a33382b613c3b1b0a7e2c96deb303d2eccd093`
- score hash: `sha256-3c5dd292adc8cbdbc0e8f4d67dacac850390302d59265cc6a086795244f400e3`
- bundle hash: `sha256-5419d26490f9caf3709ec89d83248414351df042060f582d341ef12a143f783a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81d1a7582801981771b9bc27a32c83725b8a8a67e2715cd65f17099531df2d18 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-188451956ba5e42a97a7d35f84039502e701c59ed70151e9dac58d48f5ea52a8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3d775a9947281c11e31ff33ec36d3de38a4484bcec591147d9558dc20bc3231a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-445b59e36a1bc7e7f7a2874a3dcccc94ecb6f4e4ac8fb38ac4fd3fd5777193f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4837c9ea | sha256-c3d9bf1a9a522d6967fb747da56e9b29e9760ba5098545d682839f247c8e4dcb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4837c9ea | sha256-d0dd2dd03abd7ece6640d89b8e1a4235e19211fcd2980d05a37d0ceb8256335d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c206a2a3 | sha256-df45f52fd40a0531c8277afd83b382bdaf88d6b62094213c9d196834768f763c |
