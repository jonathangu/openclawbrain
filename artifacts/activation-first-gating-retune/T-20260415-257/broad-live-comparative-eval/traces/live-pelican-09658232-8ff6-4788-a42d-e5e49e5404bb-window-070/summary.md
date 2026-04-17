# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af0623cd896f3d36aa832764b91c449eb65a56e502af4829ad2995082aa19cee`
- fixture hash: `sha256-729b2a143706d45b443dec7a409dfdba222ee805edd97aecb9fe78e30ae910a9`
- score hash: `sha256-9eebbd854695cf25697f7000c864203aff7b05cde5165226b027da05b608c211`
- bundle hash: `sha256-a7a874c7c58ab4973bcf8391920f9da603a695d1cf6890aa2cf8ff556b43139e`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eab0a33c4abd2c359ab539aaf769a166d947cff61714364e3366b892e52db8a7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-561d9cc1a7ddd2e396659130474e2d76ee83440eaf0ecb10390bd71e4cfcc655 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6f43b6aeb7ae30404cb670eeb5e93f21af1958e7b43087fb16488864bf07dabb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4051394 | sha256-d7afaaf2f7e8bcefa5cf4f7ffb57fafad2ef12f6519780de08206e62ae1ef7b0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c4051394 | sha256-b0594a44f0506cd97b2195e341b99822aa21a1da2848c5671b8a16e798de1e61 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ee6ed5a1 | sha256-af8d7cc6087c7f631a4f6fce114a182d4551e96943a066476e72f5dd3783e97a |
