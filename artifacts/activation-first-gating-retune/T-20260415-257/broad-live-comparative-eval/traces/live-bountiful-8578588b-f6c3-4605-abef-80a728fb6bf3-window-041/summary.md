# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3efd8d20cb1c9f5888880c948560443e245c94c38e7a8aed8360e4337502a229`
- fixture hash: `sha256-30c95d9f64f0e6f7685627b63849a05509d84fe08cfeea8b94bf36afddd8cab4`
- score hash: `sha256-0bbcb091c40d2608c7726d4e585cac426b8e193bef12cbc7cea5031a3956b4e7`
- bundle hash: `sha256-9978f10e4089b07202b6edca4810bcf1c3d58a27384a6152751f326ef45c618d`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af57ec3a983f1e66674a8f934086aafaefea804fe02b97ff477ffa64d3b5023d |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8768579ce6ac2a82cb64503e3e65bebdbc9ce99d15940ea8f80b931f7d998fe7 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-e32f76cc9d33f8f8e4a534c8d7fb65983ed87f4ca19503a5781a820c9aaadfab |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5c08953ea1662c9adb9960c010c22f930a853f86ed84cfa42381aabb97b4053f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-1ef29932 | sha256-5ac731f7b640f86925edf8d322736ad5f92d86b0a99a16e4e32c1e053b3eb6d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-1ef29932 | sha256-86b8907a60b474106941fa58f09bf2c502bf2730b18c1b64253807d63dacb9f8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1ef29932 | sha256-5ac731f7b640f86925edf8d322736ad5f92d86b0a99a16e4e32c1e053b3eb6d7 |
