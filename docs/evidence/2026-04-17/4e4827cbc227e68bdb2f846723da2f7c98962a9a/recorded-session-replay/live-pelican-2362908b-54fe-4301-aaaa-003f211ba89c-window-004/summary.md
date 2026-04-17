# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae1abddec00632179423e5d665c773fa81ea75d92b306fc15251840d9f53ec48`
- fixture hash: `sha256-c2c90149661c99c58bd2b000a17d70b99f16ed3daba941c64a7e5c1b67ab99b9`
- score hash: `sha256-08305782058f3b3b65cd9ee53a932236f435f0cb105fa15bf41391d5584ab9a7`
- bundle hash: `sha256-74e0f883005024e21a485bffec0ce78fd739581a057e7f05ca881d76de6f3fa2`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-83c42e700005538dba5b3a6d69c6c5e443ab91af8b598837eb4ca6b5f8135237 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3ec16a312930fa2986f748b541428e616755e4dd8c48a7ab03bf55932a6f476b |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4213fb7b2965fd84f89ced91194154d991e3f01526fb474730e7c28b0e85fc34 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-a55f45eec86a48190eb4bd4f48faf144a430b9f2508ec772e9508daa9e048f8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5cb0d013 | sha256-bffd2a4e23b1b54befe0c5ec72171cbd63e4419fa97c21e0f354ce96427ec6de |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5cb0d013 | sha256-c2c870fe2e91c429c1c233af25e90bc2bce857628a42971d0d52895593b3bd1a |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-bdc18284 | sha256-a2580005453e6615dd18c15ed654a9da69ae20e7dea8798784640b3f88f0dfbb |
