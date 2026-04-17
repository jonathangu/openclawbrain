# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-970ba48dfa6c96d0a4965b4677af4fd629fef3cbc40e01188dbcdc91cce4557b`
- fixture hash: `sha256-be39fb4084ab4014f594ecf827b8324c7590b1b3c6ba2cabd9bff2dbd9a1798b`
- score hash: `sha256-455d599ccb1380a591476a42b8cf20aad91dc0743bb33c5c6ebdac00213a92b4`
- bundle hash: `sha256-06b3ffac26c5d29544b30a7b955737815f4621a9427ec5478442404bee0f696d`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a8c8fb966bff98fd7248d900de12653a4c0149cb3145489937f87d5ed585d1fc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-717f0c84d14ad2d55bf9f0c492cbf66b08d274929fae702f25cb6e73e9e38218 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-756d3835ce12108e76a8168765309f3e1b07aa01629158c1d71da1563c668d72 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-08d425e9ee4b1d624929bf65e44a5c5c8bf8450ed608a78d6d7c3e8a882c29b6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ed9d9699 | sha256-fbde076b48ec9dbe5783bb52f4aab201f77d8dc3bb6886ac48ab9e809719e435 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ed9d9699 | sha256-d1c5a0c5957f184dbd18f5947b1692923953f8e3933132e0b3d46001fa5361e4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-89243a68 | sha256-45b6f65d46e7f4d709b7587d6330de05106767aaba4b7d714f1b79352f62a34f |
