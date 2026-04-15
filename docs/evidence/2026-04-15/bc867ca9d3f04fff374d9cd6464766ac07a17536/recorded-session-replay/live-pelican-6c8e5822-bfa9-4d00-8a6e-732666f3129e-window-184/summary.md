# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-ffd52c6edd26e485d9424e4bbcddebbac84adecca96df98ba66cd784ed30710a`
- bundle hash: `sha256-677f55c9602bee9c761e4302e6e096c8afdfcdf11907c33fe64e4cb1be810964`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2f4b660a3e5d5f4a920994b92d0eae72726c74b613f3139fcacbac22692626d |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d453ecec4cd925ea55beaa659678659d9ae298698310a0dd5a1d49cea4dd8069 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b62f534622b315d3052520e910c370091e0a89b4b12ff65bb9bd207acc1dbfa |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-1cb5f367a03e47ef49adcd182ba64eb49357e49dd012b8d6ef6f096b1e421b12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-16edf479 | sha256-800d66934267cd81bce82c7639f920c33b7689961ae836ad6b3c4e51a66f971f |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-16edf479 | sha256-53be83b52c4c9fe9df84329fb8bc6007672ca867c2a42379bcc53fcdb32d3834 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-16edf479 | sha256-800d66934267cd81bce82c7639f920c33b7689961ae836ad6b3c4e51a66f971f |
