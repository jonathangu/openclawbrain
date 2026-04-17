# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b194659082568a82c511a7152ca31a2b1b95a8940775e0d8501ad2641699262`
- fixture hash: `sha256-818172b532c3157150cdaf4f843fa921402c9f435a9b49f1a0bba05b616c0656`
- score hash: `sha256-351a608bced09cd334530c2617c71a3956be9f25a75773cb627d2e98e8cc5130`
- bundle hash: `sha256-0f8c15771edc62a6ab07dc6e7ab6aa8c33f1e7484fa2a9074bcf8920ef2d6afd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c006aa8496cda3a74dc0aceaf43d36eb374dd8330caeef238bfc730df80da87 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-63a5c72aad7649033edcabbced1b8abc86c2ddede6a428dfa177fa2f576a4df2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7876b8e4960ebde615358630de417ddfe23f728425f021131253628d4a5b5425 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4650d55d651a1651f8211bff2abd06886862dc97b2218eaf3eb21b0244c36c0a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bf14def | sha256-e3f79c2b937c3820f4f48e47b12c4c249ea77b89461b36d00781fba2c11cad2d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8bf14def | sha256-609f0cc692d209de40b01c57b5a483e0df27a35387aac1ff339e73b65da14c57 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d721ac78 | sha256-0bb85d27cefd656c6718866bd66914fb02d4cd5b0c90c921ce1c43222b3ab5bd |
