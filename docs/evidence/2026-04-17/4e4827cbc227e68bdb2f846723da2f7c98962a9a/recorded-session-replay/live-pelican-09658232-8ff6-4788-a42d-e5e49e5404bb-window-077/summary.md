# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb0ffa9f4e8d70c4fc8ddccf35cb362423daf7116804c6734e40b9d0f4296bf`
- fixture hash: `sha256-810d932d8ac4f8eed98f074f82298ad7f5b0354d5fdf19533c533df6c21240d2`
- score hash: `sha256-28b03a8b6313ee1a38b39cae906d4fe2317c2faac358100c791a4b32e1cde8f9`
- bundle hash: `sha256-9d54a1c67550d8f8382c452f9546947d057c88946b989894d94c43f14bce4a13`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe2cee15d245f859cb5315bfc802316abb7874a3bff97839e84f3440b5d4a896 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c87d5338c1f88eedae06233000e9f36ba38128696b9bd8cbcde16a15e67d2bd0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9701d5782d755c215301106a06476bf9383eb65b1c63d51bd58a268eebca8204 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0b8831824ac1a9a39a2612a54b98794cdad9c36fd748946be8d7015030232d6d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4411c8b6 | sha256-f9f93d16564a1bdb8d7d3a232020af76f0d01c1de57990925fd9b66a0b5f74ad |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4411c8b6 | sha256-43584c637d82af45d9535089d1d86ec3a5595d7f8a28983a1ace52e755f8f33d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7901d203 | sha256-7a3f607465a3daa4e8e430c9223d2192935491a51d8ffa84413658662af9c039 |
