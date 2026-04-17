# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc7aa6c27637d299d6eae706b4fc67a2a2a7b4de77c818a562317ac57ca7ac6f`
- fixture hash: `sha256-3d9a8c7638fdfa743ac7a63700e6bcceed5b6728eed1bfa78f1b2db0ab28c6de`
- score hash: `sha256-a2f16b6b5cc7dfafa00ad0f092c8f4049a8faaeb637cb1959477fe967c9b0a4b`
- bundle hash: `sha256-bcfd6a68a44804883b34330532814b09814d40a283f52f9e9cbec31166d56be3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6474375f5bcb6a5860753785382ca496af4bf19e7ca31262302583c0776eda20 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-99fd3a746d2375d480f3a648f5cdba334856269e1d42b56c824bbe6305683012 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9e46cb49469244604d33a17ff309ec0a0817ae6f463a5a7144008fab50bb9fe |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1dc8a0ca403f23d4abdf2d957102641503401382f363422a0776b3d100cec774 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b8b13b1a | sha256-912c9a9215330547ae6a6d685630ba6ad5a35ee030680c81b3c5360fb0b9e9fd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b8b13b1a | sha256-100ac63c7a9cb2f7ce92576972fe4a5943066cb13946b08aea56e8413ddb3857 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d32fab07 | sha256-bfab298f55ac140073a2740027c8ed4af398f5d2a95e08e808a082ca951e7b48 |
