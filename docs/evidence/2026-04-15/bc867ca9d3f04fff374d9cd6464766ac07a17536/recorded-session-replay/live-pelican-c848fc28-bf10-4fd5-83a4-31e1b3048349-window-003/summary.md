# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c65984dab810fcd56a73ba24f7e48a3de3329e9e72c9abc055205970cf393432`
- fixture hash: `sha256-6edadb4cb34df6bab57971cb77cafbb8b923e3e92f73e144950ce412708011f4`
- score hash: `sha256-bb042a69313f600212e0974748acf23852fd8f0738ecd9025ee8921bb520377f`
- bundle hash: `sha256-58ef6d4dcfe6dde8e5e4cda650974b64a338143c9b09792a5ac0aca03641b117`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b98041498153f3fab8845179ecda7c5ad292ef71a993f916db2031745eb7d0a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b08e0b94baba324a297f49b5e3781e5549889b6b62ea4f593c11ed41d3b66b0f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1183aab35a6f6d593b7a67520b81d9e471e7b6479a7eae089cd9653e36d5733f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5c128dcc2b3a88d6484d949538b1636106ac4d89d0a7f3dee0aae7f64170c89a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-433c03c5 | sha256-e0bfb524f15e19b2b12484eaeeeafed5c0c8810ec347a284cdb8d664ce737140 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-433c03c5 | sha256-f8f1f24a65756f95652d9358d0d3d22e1b30b3a6cc15a69506b1a395b3e95024 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-433c03c5 | sha256-e0bfb524f15e19b2b12484eaeeeafed5c0c8810ec347a284cdb8d664ce737140 |
