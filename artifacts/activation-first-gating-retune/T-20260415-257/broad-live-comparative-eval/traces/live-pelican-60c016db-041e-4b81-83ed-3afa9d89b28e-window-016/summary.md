# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3356e25999580c1815d0ae49bf40bb5a370485dd768a3aa1572de8bdcc8cba97`
- fixture hash: `sha256-0729bfe4197c5261cfcc1c8ec0f8202300a63367aac183bee3a483ca417a77a1`
- score hash: `sha256-be1c8309f4f8f733a86a56d7b5ee4291fa04d760030e451e817700efad79ab70`
- bundle hash: `sha256-eb0508b1b0d61a3f24fc9674b99788bda8ca0ee45d8247c77dc76932beb8bbfd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4b42315dd1b1a176628a7fe8f13cc2dabd4068509144906d5e0b01e2bdfcba3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98e31302b3df75aa1d0dc7fcab38f514576612fce9e0d893e461a94e36f21f24 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af3241fd1ad427737e795c57f3f933990ccb236deed44366070be4fb0d318617 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f13d454b1f1e18447aaa669ad35546e74ead1bbd3786cfd98485fc2d9e8369ff |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9b9037d | sha256-7eef2acc28d90362d68662857156b6203156e79614e6761a75905a30e62337e2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9b9037d | sha256-ff85ae8e869719b4c9df6da6a223cc6768138c6c468383dd4fbb89d6e63e0a58 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-45d0c810 | sha256-68f6cf32a90f5e0f1bddba7157b2edcd1527ce8410557c3ee9134cabfec5cdaf |
