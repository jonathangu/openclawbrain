# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4b7823ea-a7a7-42bb-b79e-cefdbc1b56ac-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dfa81f4d2b00217c5c5c520178573740e8780c6997e7fbd463fe714331cc7869`
- fixture hash: `sha256-ed00dcfbed6598ace12042db40479b3199c9a2955a7a673a786b8d8fa048ed17`
- score hash: `sha256-aa6acafd9a341dbc99301fcb2408d45b9d9bcab02f3f67b63de2fbe3cf38c421`
- bundle hash: `sha256-cf7931001f966baaf39d37c9f1fd7c141087b8afad635224e020e860d733e431`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2d5028651743004fc65c4abb7a18a3ce781f93f13bd67703dbd698c51e61ae2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fbd27e641c2219541033b8ac482e970a6f6d3387056ead6a10553251afcd27f2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7cb429e28775a6cbc3ff066cfe364d4950c0d00d49b987aeed2deabeea6235a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ade02ecd18b45fb377da20f641a990927cea31e7bb6f78188807a10f11f6022b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8a928bd3 | sha256-249a35ad754d9557a4bd7f1bfd4b8e3a0b7add5ccbd6b7c7b77ba003c92ade4e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8a928bd3 | sha256-249a35ad754d9557a4bd7f1bfd4b8e3a0b7add5ccbd6b7c7b77ba003c92ade4e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5cc9dcf2 | sha256-1029d0070d6a97eb8e1744b57dfe5eaf79962ffe62738953b64b4cd012a461e6 |
