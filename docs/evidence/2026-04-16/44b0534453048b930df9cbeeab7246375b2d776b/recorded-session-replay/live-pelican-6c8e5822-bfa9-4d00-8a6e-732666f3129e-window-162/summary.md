# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b02e3d7c43b0542a9708c97a4decb5ab50a7fecdb19a413e8ba04a6c6f24587b`
- fixture hash: `sha256-fc0fa875ed0ba10ef61e5e8b6c1b783878d38dd1c5525b62b1d2717e4e66617b`
- score hash: `sha256-5584e54d62adba38ef4d4b5fdce83005fa004b6f88d9c85211d232ab0a2ff7a0`
- bundle hash: `sha256-c194156639ad0c0f5f1476eb1bb6bd1aa7f401d31df7cea9ebe33d03026a72dd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f8f3d7baac7ea624c59c2785d2ad8b5f8904cda6bfe17f914b150feacd473265 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1abd8a96205718dddeef98f4719e38a772cd92f7df42c80fc1beb8bb50a668c1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0364d1334d732316273fa46733a806f1785e9bd00cfb26d48e79e901daf6afcd |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c42ebdbd7d9d3663c0ab1eb1b3e8031d1b473a103abcec5371f3640f2b7b9af1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba4d03d3 | sha256-3e1b635e1bbc62c2d61511e204f609804534810799c33c28d83427d833fde7e3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba4d03d3 | sha256-dfeb5781e7f922fbab75478fae55b2ed28f57ec6f69abc8342f037e189e596b0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-12b64980 | sha256-280110a36c267bbcd61968528a84181718f245a09ee90b38f32a0dc855c22877 |
