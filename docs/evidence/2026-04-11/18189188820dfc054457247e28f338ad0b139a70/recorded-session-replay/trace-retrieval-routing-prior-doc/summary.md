# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-routing-prior-doc`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f541ef591a3fdeca44ec33d94d63586bddb7736c330abf7d4032054bf4d0e239`
- fixture hash: `sha256-10d2f6679e370351117cc6864a5d47c336bab40268450d59ff126a9b849b1bc2`
- score hash: `sha256-750d3084eb45cbe04978ace42223446c0d2ba799bcabb9eb23015ad019a82ed5`
- bundle hash: `sha256-620c29751e10aa5984d4844e22483008ceaf9073c5c58537ed741e0a5caebdf7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-05be127f6e13c4988ff9e6455aa8abdfef90f76ab4fa10fd101198d4fa2025d5 |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-34547a0bc90ff9a7ce9ce0a8873c3fabcc07f7bef965c00b6430be3ded80a2f8 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-37753b81b95bd405108c8d46b23f4416ec0cedea0fccf31357e8fec4a5c7f919 |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-3ba1994b936fb45c8a1b0f728a1acdb96c2e0530ae93033584b3acb349fb6bf7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | routing-docs-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | routing-docs-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | routing-docs-turn-1 | 100 | yes | 1/1 | no | no | pack-39c270ec | sha256-54774982757b73e99fb2dc8736acab225b689e7dc89ac408230d5974d984047c |
| vector_only | routing-docs-turn-2 | 100 | yes | 2/2 | no | no | pack-39c270ec | sha256-54774982757b73e99fb2dc8736acab225b689e7dc89ac408230d5974d984047c |
| graph_prior_only | routing-docs-turn-1 | 100 | yes | 1/1 | no | no | pack-39c270ec | sha256-54774982757b73e99fb2dc8736acab225b689e7dc89ac408230d5974d984047c |
| graph_prior_only | routing-docs-turn-2 | 100 | yes | 2/2 | no | no | pack-39c270ec | sha256-54774982757b73e99fb2dc8736acab225b689e7dc89ac408230d5974d984047c |
| learned_route | routing-docs-turn-1 | 100 | yes | 1/1 | no | yes | pack-39c270ec | sha256-54774982757b73e99fb2dc8736acab225b689e7dc89ac408230d5974d984047c |
| learned_route | routing-docs-turn-2 | 100 | yes | 2/2 | yes | no | pack-c2b8067a | sha256-051f7dd3505dc20e03a64dad71dbeae029538bd258338811a009770a2a4778a0 |
