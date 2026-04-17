# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7ce9d77f8d5b34f5d4a2ff238035837b9a17936c8718ddcd44e0135af5ed67b2`
- fixture hash: `sha256-b278e1b6b555771ff403bacda1c9f56aa4593110af14f3b45502af98316b55cf`
- score hash: `sha256-0830f2bc5d6ced65d7e210dcd7f19cbbfdae92e1b8bf7ce5982a1a3e3e04a6c1`
- bundle hash: `sha256-f382f9de7448f20a7a05bc40d357afcfcb06cd590c007a18796b6e3d3565c374`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fde92dc9f75c3ed7b5cdbc92af57a8fdea90f988cee9df5a6592eb109fc517c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4d224d5b4490266fcc196e0f5bf27d2219fa22ae2bfc4405f95e4344f21188f5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94db277b3a5f49e9ea6f6f585a26eed208f23802d0cd41fd405fb9acb45c1721 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ce59d766e5d62b1bbc844b0d4892a0cae173f60496f61c5e756635f55bf4d914 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2d70960b | sha256-8e565d8ee95962a6bc595dd9728aeafe521d014f34f18197c330dea455a54cee |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2d70960b | sha256-e6418cacdbeb4c15dfff854a7f93852182b45f76a40fbe1088064ca43aa9b90e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f466e0d8 | sha256-d237511cd433c6f7f47493cd6f28fe65ecd80cd82903101d6f747fc4a70e308c |
