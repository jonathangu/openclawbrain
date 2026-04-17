# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80f4bd70b8f229336838d17a92921bfca64745f162a9177679361b11e355a256`
- fixture hash: `sha256-1a25c630a19d83ff4b3784d9a97b879e228a965826872ffe6bcc2e6453fbac5a`
- score hash: `sha256-7fa5e122b988b216a95ec7d819cb91a127abed174622db249d64637cc6b844f2`
- bundle hash: `sha256-f2b3804ab34de1f9f15cd61f69abbd4bfff2a4efb8fa48956fd0b4e2e3e14d4b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94ade431a5c986254405c71afb4d4071b897c04f7cbfe57133fcaa9500ad06d1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8a1a5781e0eb187a74942ae83fc104255dcca17c5d6698b2b506bb342a4bb2fc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9e1db6aab1354f50c01df4e1f24f7ecd3224dbbc1debb971c44657c8cbfcec9f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f373ee730e24e9383fac208a206a5469f42e6481dc035b3035b0e43d57025441 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34620a4d | sha256-2d69799fffc57484730879c4908d97696c6140fa9838d9f6c02f70523277c3a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34620a4d | sha256-a652afc834998b72ea64e156b4b586056a8c89a17a5ae068769058a6dd309677 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-643bc4a0 | sha256-9bda4b9b1310390aa6ea9a189a4b6a4604cb13229ee4af53e18c9e7325232a20 |
