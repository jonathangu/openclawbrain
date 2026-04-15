# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-6a1408c2399981ed23f3719871b17549667f8d158343d6a57358dc5740bec41b`
- bundle hash: `sha256-c1cb0f527513a444d56705321cb05d74b20a4ca2c1f49c09d9a97480f3e1ab01`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-76367337345286c54e0467369a7c393f6c7d81360b7b381c0b6aebeb8349a5fa |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-2cf7fb295fec924ae719d106919d0e896fa38810bde27aaed703ee0b3d59a0a4 |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-bb8340b42a42ae1ef121598ddbd05d860ef8a2324f492e678d9f7fc10031ea97 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-b114a2dd | sha256-3bae6692d259d8e2af5d67a270362361b208e26f8991f2580afb79a673794d14 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-b114a2dd | sha256-6c446f44d8fe6f0e59188732140131b40c784c9a4ceee33dad8d70a18317892e |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-b114a2dd | sha256-3bae6692d259d8e2af5d67a270362361b208e26f8991f2580afb79a673794d14 |
