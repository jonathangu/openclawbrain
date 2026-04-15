# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-773694158527e8c12eff6161a1badd3c1803ff909a44c706ba218a63cb990600`
- bundle hash: `sha256-b7a51ec984d628a75555c900166a79776422071ada91337ef47076d8e71a1f1e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-78130c2e9fdaa903cb767bc9926914f1754772de3663c2295eda3f240c588b8c |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc0cfaced674ef34ebba41f889091cef1054eb2848e87989fe0f4ba02dbeb2cc |
| learned_route | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 2 | sha256-3f1899c83bb6c3e3a4b74c09c5c596ab473f5ee2a84bc0a1007763b170cbb129 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-5a07cf42 | sha256-da73442b833ae28610d3a016f46d91b7b0ca776161c914876b4e4b9166d84996 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-5a07cf42 | sha256-e09b9f8bfa428c568d3e6575397890733c143c4435c1ed73f328a1f714c2f6c2 |
| learned_route | turn-1 | 100 | yes | 3/3 | no | no | pack-5a07cf42 | sha256-da73442b833ae28610d3a016f46d91b7b0ca776161c914876b4e4b9166d84996 |
