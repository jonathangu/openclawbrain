# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1daacbdd680cf4033ed5d9fa2efa105e6544ffe1129c7600ff85b76c0c2f8393`
- fixture hash: `sha256-ef7e749ef838de36d236aa29e0590a88a86c6b42be12cb84bf00123ad9c263a6`
- score hash: `sha256-08d9e6048ad2d08b1fa5e8ed9103ac2aa2a0c882b7b5f0008ccdbf683c378b4a`
- bundle hash: `sha256-87a4330e994e02ac9e3a99ff3773c24bfb4bcc2d3091aca4f300228378a7912e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-715391ab03706266e4dd92a9d6ff099345f003fb7379bf779cf731b9d18a7950 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4027810ebf9f05a5ca12071ec8835a9cfcd87bd5a716a99eb7ee5112cafe32d1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-969526ce3204f5787f7e4f6839b4a8d587ee68526c39ece67f2502080c397e5a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5a0cf037295c54b28e7beca1268b2c3f5647238e9a8defd1c3c6701244797606 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-849d4d10 | sha256-0a76cbd3c2b2c0ed4dfeec5f6aee1e0d24447b6838badadbe3709a740d7d293f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-849d4d10 | sha256-4550e4c74379557815c69f8d175ba39b39ce8b67e938ce0e8f23b3ecdd4bba3c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a5f95533 | sha256-583b12e74425fbb338dce7cdd552859f8f8417977a56e3cbb9589d050829c2d0 |
