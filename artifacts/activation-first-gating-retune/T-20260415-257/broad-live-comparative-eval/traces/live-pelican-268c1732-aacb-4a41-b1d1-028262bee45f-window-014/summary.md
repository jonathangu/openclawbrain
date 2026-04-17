# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-82b94292f904129190996d09645352442519cd34f4a6fe4ddc3d8ccfdc15ed4f`
- fixture hash: `sha256-2b7971a9291be722d620678727dc2afe570e5b9dc9a97d0983cbb8375a8b4f0f`
- score hash: `sha256-041d65ff3a98d4ea39aba8f4bfc86bc7f01160404961e2a86e8c2fd16e507c8c`
- bundle hash: `sha256-9f4cc718843dee0fab4c8a6e648a9531a7451f421beab9c62cb9c10110d77869`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2126389181abd46124f339c97d016b2e80dbdd1c3f4a30cb14b5104924e09f3e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8aba029e4bf29117ec9b3ccec8e29d517e97e4239527269d659b8bd11e4a26dd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c46ef6104617dd71d87fa4e9706c912639ebcf275e380e811158c6de03be86c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-392106eadfd2fc906300435c6aa2be51f80b698183279d2403e9317487a8e1c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1c204c3c | sha256-1cdc64816cff4ecaa9672e5f007925a3e4df7009918186abefd0583e40a35d1b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1c204c3c | sha256-e1c1373740c48f163f8d9c115b92689c80eaa964ba2ea5aa1e343be16ab589e4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b0f35619 | sha256-eed21eec7f132cd12cb1bf1dde4fbd8eaacd61252f285093e290cec52190b9b0 |
