# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b194659082568a82c511a7152ca31a2b1b95a8940775e0d8501ad2641699262`
- fixture hash: `sha256-818172b532c3157150cdaf4f843fa921402c9f435a9b49f1a0bba05b616c0656`
- score hash: `sha256-62d577cf3495dcaa3a9688d358d7238f39c5082fb6bf939597407ade941a838d`
- bundle hash: `sha256-fcc35086c064ecaa84b52c98ace0e00e337bd49216f682a2d1e8f3e05f547a54`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c006aa8496cda3a74dc0aceaf43d36eb374dd8330caeef238bfc730df80da87 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6752611e2fd047b86d9989666466bb9685a9b13efe3a2107568188f58f2c2b39 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-06e0afba9917fd69f252adb5661f6f686168b68ecaaa7aa0d8a2cb06b0b91d9b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-9ada2be93eb86694e28d549e951758b1aa84f7e0ead2238410f629d6738220f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c4e63ca2 | sha256-922271c44cd67d5e4f67862d2510a3adcdb11240e81f0591fd9dcaa417643952 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c4e63ca2 | sha256-d88b12650c71f0caf6021b87d46abfa87a10f20aec61cb01dd75ab2e02acaa87 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-c4e63ca2 | sha256-922271c44cd67d5e4f67862d2510a3adcdb11240e81f0591fd9dcaa417643952 |
