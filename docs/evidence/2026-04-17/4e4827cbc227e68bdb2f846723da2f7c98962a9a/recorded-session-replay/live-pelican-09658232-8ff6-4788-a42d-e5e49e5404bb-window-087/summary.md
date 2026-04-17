# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bc492042fe348d82c21faa673d938d0577346ee06614b38f34d614d883fe125`
- fixture hash: `sha256-166379a9c9e98e60de3e148d45fed20846d7dac8b779bfce9e0299ba405d4f98`
- score hash: `sha256-e10653cb27f00b494a84bd09e4a37e01caca971c5b4fde310611db84ec87f299`
- bundle hash: `sha256-e83ce5d6fd722726b0d4ce37b324a641352395d009b2843f08926972c1af8cea`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df7544610c7c12f9cdd0d8aad84f983991755b60031939cec1112c0295581782 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-53319f7597cf8245c33078520420db77b89e1b8e499fe2bd088e80f49885a60c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4446a9b9c94fb3bc424008be467ac5954c3cb69e91566737ce1870370e431afc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-807fc3b00ba3edf05bf0adfddea4b14117d8511f9ef275593aa9b0d0d175c87b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6291b4cc | sha256-aeab7f05e9de51d6a1556e2900fae31874b4d7896ef7b83a2b7d275da66797ed |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6291b4cc | sha256-8de481d5cc32e6f7b2ba1ab57d01842ba2e8c2d6348691ec08afb6a38e0df06d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-25e76729 | sha256-881c15a02622c3b947b67c2e9038dc8452b31c77d53dd6b4962e150c1672eb4f |
