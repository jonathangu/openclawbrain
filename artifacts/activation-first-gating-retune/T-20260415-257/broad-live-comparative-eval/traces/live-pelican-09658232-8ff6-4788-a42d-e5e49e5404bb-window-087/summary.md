# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bc492042fe348d82c21faa673d938d0577346ee06614b38f34d614d883fe125`
- fixture hash: `sha256-166379a9c9e98e60de3e148d45fed20846d7dac8b779bfce9e0299ba405d4f98`
- score hash: `sha256-4c779bf3683bbc3e9b2260068174edba4b69529ac7ac83a19f78243eab43919d`
- bundle hash: `sha256-80319afc3e04339015d15a88e4ffee980efe786050acb5ca550102df73d07fff`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df7544610c7c12f9cdd0d8aad84f983991755b60031939cec1112c0295581782 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9d5a999ed248e8b69a1d82eff709895a8524c612a24d3e9a4ad03b005e16ad7e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-753f6af77b6a27e031e1588a47b9286386883851a04f789db34d4e67e3ea05fc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-39fb8f8c107aa261067651764527982ed711e4e64498f1971d3c460f516f3790 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-399cb6f5 | sha256-0ae2a55978e41a22e349fcd508847a5263cacada8f16bf1996f9e26d571d8ac1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-399cb6f5 | sha256-5ea2c26ecf9b3e4f458793de2da2c6b3d35b7adf83e1f382bd4a29a67b7b193f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-fcf26952 | sha256-ddb4a9287cb71c2d7df4320d08ed560339b3241ca22e0c4a528a8ae83f597856 |
