# Recorded Session Replay Proof Bundle

- trace id: `real-trace-crabpath-handoff`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c9984f5c500fa074515651de0386d2fc31c37eebf193ce1bb84b4e3d0726d450`
- fixture hash: `sha256-af19b6e825342ce2ba90aebb643f8d7dc5c6fe3ccbf79847e338294013ff839b`
- score hash: `sha256-5e5a21c76856b090ae45afc70f163c9a7e09450474e40ef1a4181dffdd31ffae`
- bundle hash: `sha256-07e27265a0b515135e72ed4614ef0e9b895b1227cc3c13ef33cce64fcb52e616`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 7/16
- phrase hit rate: 0.4375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 0.75 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-7dd3b2db9aa8ec9fc17136da05597f581a47591599d10e6947135a1e7d69fab9 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-8b60a224026e69fe7d922ed0b4706b7c84fea331cfd8a2ae255e6bbcf2b8a84a |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-f684aecbb34d7d7075c862392456f49b6dc78a5b3774280b192dc7603602b239 |
| learned_route | 3 | 3 | 3/4 | 2 | 2 | 3 | 2 | 0 | sha256-39a7a67c094c14c8cd562bc23d0a1fb71f5a8f4f7497d54043d596fd1b7e64bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | crab-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | crab-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | crab-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | crab-turn-1 | 100 | yes | 1/1 | no | no | pack-f9a5a38d | sha256-f9b396963b9e059bdcc488bf2ec868d3467f263d0e0676ec3966bdcd0307873a |
| vector_only | crab-turn-2 | 100 | yes | 1/1 | no | no | pack-f9a5a38d | sha256-5c173f606d32237da206ec054e84a9c216d4b3c93654e4b70b38011361abda81 |
| vector_only | crab-turn-3 | 40 | yes | 0/2 | no | no | pack-f9a5a38d | sha256-f9b396963b9e059bdcc488bf2ec868d3467f263d0e0676ec3966bdcd0307873a |
| graph_prior_only | crab-turn-1 | 100 | yes | 1/1 | no | no | pack-f9a5a38d | sha256-f9b396963b9e059bdcc488bf2ec868d3467f263d0e0676ec3966bdcd0307873a |
| graph_prior_only | crab-turn-2 | 100 | yes | 1/1 | no | no | pack-f9a5a38d | sha256-5c173f606d32237da206ec054e84a9c216d4b3c93654e4b70b38011361abda81 |
| graph_prior_only | crab-turn-3 | 40 | yes | 0/2 | no | no | pack-f9a5a38d | sha256-f9b396963b9e059bdcc488bf2ec868d3467f263d0e0676ec3966bdcd0307873a |
| learned_route | crab-turn-1 | 100 | yes | 1/1 | no | yes | pack-f9a5a38d | sha256-f9b396963b9e059bdcc488bf2ec868d3467f263d0e0676ec3966bdcd0307873a |
| learned_route | crab-turn-2 | 40 | yes | 0/1 | yes | yes | pack-30496916 | sha256-e392a416ca968465ff7ae52496a5f716952d37747725279409cf94ce1f7cbf95 |
| learned_route | crab-turn-3 | 100 | yes | 2/2 | yes | no | pack-6ea7af5f | sha256-47ccc5229dc361edafb8c6f89e27e8299a53869166b1787c0674c13d0a864dc0 |
