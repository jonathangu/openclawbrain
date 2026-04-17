# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f56abe6ae0bc3db0256fb8a29def5410559dc260bebf6c25f482767c44ad563d`
- fixture hash: `sha256-ce8c7634db2cf81b1d292959075126cdfc5e0bd2ab016e812a771d5225b9272a`
- score hash: `sha256-210f15ab0f6a83a62785b6752d00369ccf712bb52db29ac768de4b45cb5c259e`
- bundle hash: `sha256-794df61e1e919ff269ef1209e303ba1b52a7223f2e8e684975ff5d22758aa6dd`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0d6d335f3ffcc4b2fdd753f294bfd27ffc13dfaf4edadcd54e21edf30efde50 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c842f5b9ffd420f47787c705c7218736ef1609828cb3d9e9d33db352b783fbbe |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8d80eb61a3a714c80a3835bfaddd91dae9ef112d780e04849eba6a41d0d2712a |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-0a948db662587ad1623d37ac6f1b8024b1bb2ba86ca0ca3d1c6d79d1b9f3062c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-34443937 | sha256-2e44873f9df5d93ba4cb36e48b107b04c9007d03a2e907de3c1e3f536e786069 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-34443937 | sha256-b91c4549f7a741807955c58b826dd544df646262f0483ab8f3c15cde0f904912 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-6de024f0 | sha256-0198929443c89eb2840db7d2263a8f96da812593d1fec2760e84d575f9896c44 |
