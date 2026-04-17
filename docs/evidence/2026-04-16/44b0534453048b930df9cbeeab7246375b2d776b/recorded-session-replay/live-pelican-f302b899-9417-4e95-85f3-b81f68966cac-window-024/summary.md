# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b224dc602b7429463a9b2fd5346afa6d3382bb3fd84bc9d3cceb0d3ff24896dc`
- fixture hash: `sha256-493fd471e0bb608979cd024ca51b9104b86ec7063e95845a4d6e7076002d21f4`
- score hash: `sha256-840d26a7287400e4c32c9fc7729fd1c0b1f89cbb7fb0bf84fe984bb8516f7bf8`
- bundle hash: `sha256-b8046850b8c61ab601826ae42240bd95ec9063030a68d11ac480034ffdaedbf2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff276f984ca7449fbf40ed52f8c73e2aedf05be900e45cdc0a8a0b8a46668591 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-15d9f091c468fb72c1d4030ff0bca286ca4c274f489f81f606ffe645bb54b392 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1b4d98dd33d447243a20861366302b09484617c5abb631ee81edabbf4f226048 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7ae78f6ca29ad972e794ef52ae48ee1f8d8ca0ac497b0b63d365409eb6f61959 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4260d114 | sha256-ac7bb7da22a7d642e635bf4945561a39104ff1c8bfcd33a82217f97dccac1f15 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4260d114 | sha256-f1d67a54bbd3bc270681f0c884e20d434d32e31e7d16c6994dbaefb8553c7366 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-53795fb9 | sha256-ef4149f999aeff562c5015a084ec3a5df4191ad49a25b4e3ca3241f53d219b97 |
