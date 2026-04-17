# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055`
- winner mode: `graph_prior_only`
- trace hash: `sha256-069c659e483d79099c9522902169a0e3c008a2a3a1a608f281e5842abe60c793`
- fixture hash: `sha256-2053a334b00cb8986b08e94b050daa206b7253e27c1f42496d3a7ffe4c19e5d6`
- score hash: `sha256-652abb79b78ef7b3265b0544934c1c6845e61004b490837dbefef8d5db4157aa`
- bundle hash: `sha256-e574e8d85a125375ab830c0c92f2882ab26bf3d9e46638c0e4b2ddcdf985a7fd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7d0a9be6adcdff721e255d979e1bead77026cd16f5da5ab306eca424cee158d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c2c84393fa6e70b3e7022c540b57119cc55087afa9e2be3bf5312b62aadac578 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c922b6d88b305586ad4f9bffa6e729667e850c5f94c1b3814abf21276971c9b6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2f14ae3f5d3dbb1716dcb885e7a21cb61f1d3d6f43b2f1956be023649dd45161 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-67964b2b | sha256-f8c3e193767f83c930e279bd75ccca0613f89bbb986efba717b62a5b4945571b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-67964b2b | sha256-bd2a1c1d7d9bf003f74268e2e86bed358cf58740f7f79add37413c6341c1fc57 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-caa6eb1e | sha256-3089c69a2ccb217d26aa61cc8933231fde1f63f80e68602655b16ba90a97889a |
