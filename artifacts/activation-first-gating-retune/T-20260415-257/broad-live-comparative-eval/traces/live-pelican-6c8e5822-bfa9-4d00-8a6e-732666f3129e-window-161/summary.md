# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b83c4e3036cd20627db9b3b691867f0c4cf67798691db96c05537d9efc454f0`
- fixture hash: `sha256-d883ca17da8d181a1200f08513acd619f27d5b75e1c49c4953044231381c83cd`
- score hash: `sha256-39b14b15249f938f9794cd3ca59e763b073427559e757bd115f506f3ffffcaf0`
- bundle hash: `sha256-f1002b6f0ba029be66a26fdc2714376c7d0402cc981ef2afe1c80b0ca9ba67a5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5551839106976537fe1c9ce0dbda883b66824b3f67f3049bc2763f475be1647 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ebb8d2e1f8e6363ff0186962cd92e5a641c04f53bb8d3663fd1dabd18c18e49 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ec533c2c13adf06d22e562999e5f8ec968c5ecaff8388e3b7ca078d6cad5e54 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ee30587e18eff6eba8e87f772cecaeb0ff8dc19a84e916d0ccec2e17e1c3fb71 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aafb9bea | sha256-6cec7d5a35f10f89a5453d57bbeebce7dc856732a070b1567375e505ddce3dd9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aafb9bea | sha256-53a950a3768ab78392bb5441cb1fac03e7c883f5f913f131fd61cdd615a5c651 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bcaf3f43 | sha256-ba0c1c6a4fae3842bbaf0a2aeed8818a1f8038e8482bc3f2e8ceddc287d9e14b |
