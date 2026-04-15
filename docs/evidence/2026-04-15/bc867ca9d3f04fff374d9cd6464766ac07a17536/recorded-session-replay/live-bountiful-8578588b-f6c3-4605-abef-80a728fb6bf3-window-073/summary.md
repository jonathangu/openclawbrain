# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ea104683f4faf6c43d1e2de1d4eaf420a688cff5a862d0f10bcf59142dc68752`
- fixture hash: `sha256-f88768ec722911fceeee6af7386980436f1947b9771d5b78128543260a9fd9c9`
- score hash: `sha256-89ba1383e500db62b6c01db4f08a494435d1de34787f74c0a4c68ee6842001f7`
- bundle hash: `sha256-74bc830b6eb6097e8a659d7b76960256fd75628308904989fbc8a2895ead4b74`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-01f110226bcb696f2d92c204531dd80bffd0df5e206ead0910da9d2b251a70cb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f6991d6359eba54bef3bd04330568023af8572d52a45d6a9c678c266d4dab7cd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f2fae6d38af6a1fb3096c1c51f45872d283b83016c0ae5efb475c98c84cbf9c2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ec35334ffcf7d7ce822c31e15bb226225cf913fb95d2aa46fbfc1c6774ea4d79 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3b94861d | sha256-0c339a9adc06d7b279048a760b43a30ea77e975e2d567475c30d8e510fcff024 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3b94861d | sha256-ec89acbb74f1fa16d28c70f4c8a6602de199f0085621f409264aa538fa65eb52 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3b94861d | sha256-0c339a9adc06d7b279048a760b43a30ea77e975e2d567475c30d8e510fcff024 |
