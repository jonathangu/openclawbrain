# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fc1409b104d617856751474f01593056b66d1b2ca492e8f5dd879839efd10f66`
- fixture hash: `sha256-8310747322d42de0fb2d06597a429aa5eb75a2026f88cf3e458dadef80911084`
- score hash: `sha256-5fac10ba29855523d8e3f133b903ca62a699b0a745882e5ce0db5c7d87a6c767`
- bundle hash: `sha256-77c86fcc2f5cbfeb2e500d333ec94dcbf7e59fe5f6b79e75f860416dd1eb094f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fcb97b42b2b441ec8190e1bb06fb82b8bdd1457d8fd6d8d105b2684066c5870 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98584b4f618ac651a7a049954cf0f53eefd50c7a8ad7012aecf5c9c6f67ba999 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec6c0c86d94bad167ea1aa0da4647ebdd7154c9a412cfdc404a53b9298da1413 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8a3ad9a8941676b6e475b5d1cd41514a690d6058fc9f07ef6e503586141d3cca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fb241c66 | sha256-e32f6a23ebe35655b0f16cd6877ba1d60dddb16daed4433e74cf58286feb6562 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fb241c66 | sha256-83e79f4b2ed0a18ae2c6f8e33336385170926e1b4e4f5024e50220a0644ab3e3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9e24fc77 | sha256-682b750943b68a9803443bb28403acab6aee9e9f0f46517e089df3919f032e27 |
