# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d88b28d725d828ce19828c993023f4bb29218bcb84a66644da59b76b2ba63c9d`
- fixture hash: `sha256-87338b3d5a752854c0bedd7b04604892d7d56176980cbe837350f3ba996c423a`
- score hash: `sha256-909d173ae2ae2cbe47ca41e5098e6d2d6a8530069600c7cc7f89746bb2668b6f`
- bundle hash: `sha256-a0bdf2cb5a53979463e025f3790f2779d61fbcc60333ed303bebecf4c78c3934`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8865b280477f916340cca3eb893b61dc8525b802f5a2d26079af29780e8fa757 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5bf38be968ddb554c468ce45ea67c565117c012109b4583c35420f979aefc85f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10b6782f3252381ae44fc82b57bd302999e2f7e970369e280dc432bce1b62e6e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5e35b1cab84ce77b88dcca20f79a24044cafd38fa21b35b1059c77a3d2e40319 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3f077fd7 | sha256-0229954afb4c5227d5108c654f7b4dce773cd7efbcc8310bde30b0d925ad8c3b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3f077fd7 | sha256-f322f867766a5765bc787d64c0beb4e541ebba0e3b4ca3a55c5b251650a552f8 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ca0aa200 | sha256-8d4fed80fa8d5959589764c8fba9d656cbfb0a6d41c13f9a1102a711ed45c469 |
