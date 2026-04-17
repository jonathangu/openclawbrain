# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42dd5ae1fbc52ab37ab26b7eff707ccc814072dbaeb4cf80246f57beb5474c7c`
- fixture hash: `sha256-dae1773b38ede59c62f735546227926063dcc22433a680794834acb15197b82c`
- score hash: `sha256-a04907ebea8ac5d2aa12e27f256b460b25d0b8730e0cf5071e382195def25d14`
- bundle hash: `sha256-f9ac05c18e645859e7e69b67924a7c379f3a17dd679e05460ee2fdb9dec22563`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97339b57eb3bff564bd492b91102981f9054b332cee78d9338b804ad8b646434 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e4cdb72a5442dcc126ddb3fbf4d7732a0809cb65d6ec6a9ff19666ed236c7054 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5ae8d5a5deb09c8125071ceb46ca2850d64f81c67bc93de6616c1369239b409b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5056f85f22cd8336abc80367ee7555873f82ac9a0fa8f92439648c590c87bb5a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2db3cb91 | sha256-57bdcbc4d2e55458a77cef29eb2f2713f69499f3fc1216625912a1bb6e97ac46 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2db3cb91 | sha256-e3bf57039adec7698afa2a0d104db65bbeb3d60062b5fa4e6405379c898d51ea |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-67ef63fa | sha256-df0551d0dc5634f546e2c90cc5b8e72d7ef6b8e9346cd87a79e58e6bb694926e |
