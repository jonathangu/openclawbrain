# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-791d7ab7e1a39f9248affb0ec38376441778feefca919fa6d5dc852b64c0c740`
- fixture hash: `sha256-fc8733ed1be81b69ef5447410e17ee0e67ec342cb6d0c7a27eab065d2955bafe`
- score hash: `sha256-7c1c0667b46ae8b5a2d2aa7a0bfc4a0a7632c97293016d708ad8772e44910013`
- bundle hash: `sha256-5c1d0f592e135a590d425eb1c5b8b9b70891e20fb7658994562e0d736106f8b0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec63dafa723fc5eeeb737d59bb6d87f1a6423a6bfa624fac5dd61b64e8a7a79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a14ba7f9b2f18fe8a95fa76455be747c4aedef6b678b340b8be9343a1dc377a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aa6d7670e94bd689bea8b882f7dfb47e2e99e45f82c04eccaee064f039c87456 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-14256eb589e951b1342b9f70d9287667408af43b94f72ab4ce09e631495e73f8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e2988c8 | sha256-5c31a8df082f450ecace164f0659c650022732e3ddc419c35baa4c229bc2af1a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e2988c8 | sha256-1233d2966a008831ff085abcb6d3ff68e0ac74229faf59aa4683afcae78d5a2d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6e2988c8 | sha256-5c31a8df082f450ecace164f0659c650022732e3ddc419c35baa4c229bc2af1a |
