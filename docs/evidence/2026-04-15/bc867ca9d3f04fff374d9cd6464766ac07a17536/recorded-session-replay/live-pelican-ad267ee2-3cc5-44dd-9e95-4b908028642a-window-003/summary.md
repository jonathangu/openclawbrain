# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-154227e12deeada99188001de1f98c7859b44b0240a0a63280198f0600727836`
- fixture hash: `sha256-141e98c67b76e6b544c136b2dc9ec311316dae947241f48af13f9b3f509e9c48`
- score hash: `sha256-bc40c45dc1bb94662756f889a1191bd30e3df6fbb8e27b0240becce0b9634e7b`
- bundle hash: `sha256-3fa9f9ca476988c4e4bddc4edaefe8dd42a53ee1007b8da821a3255dcf2c0937`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd20c45ec549a50541ad825ca2263c2905bab11bd8f991e3eba1789bd6eddad |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8935395a8601672198cfda1b83a3c2ec4641f22619570657101306af5124e93f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9c3d3a750b9771015a3ce681dfa3398d0ccdd76c0652d5ea503abe5d56a9f81 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-33a616852d8f26c2b46366eb7eb4a3fe3419948c2b210bf81de997330653fce8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6675a26 | sha256-6aca45988cdc631101c9c7b9d4e902d8ec343a27e8682c4b23f5158432f846a9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6675a26 | sha256-af3835ff0fee6b420478c0f73cf74ded6fc86a5e9bcae2a91df83656ccbb760d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e6675a26 | sha256-6aca45988cdc631101c9c7b9d4e902d8ec343a27e8682c4b23f5158432f846a9 |
