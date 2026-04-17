# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004`
- winner mode: `learned_route`
- trace hash: `sha256-414029967a4dfaeacf3048f9cc246c927617fc5206e50ba6c1c2944d9dd8d93c`
- fixture hash: `sha256-2f96d4d80b85de0482bdf816d900c02ecf0137642687879ce902112bb8056ccc`
- score hash: `sha256-69b3b44d92cc4556d849ad24efff196ed33a6126da4cd30771b91c4be0e8c5e1`
- bundle hash: `sha256-85ade81fba8fcb1397a0a0041d06c7bf9b34ca7bb4dcaf7beb83ca5dd7af48a7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d02c67e07266eced41424ec8d8650df73f7c0173cd9e14609381c09dbbd89d1f |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c218a87aaee078314172d7b2788ab3b900263049d195e1d35b8cacdafa7c9a91 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7b92b07138f800cb1a93394ff73bb38350af878a13f4bb729ee6df8bf3157f2e |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-f9e08ffdb3e585eaf32c3ca5d46ddd658c1f8d416948bd7866628f071c10ae1c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-141f3322 | sha256-c457899e479541069fd546f89eb02a4412a0f2f0030fa125b028e894a7a3d0c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-141f3322 | sha256-b3c5917e9b7839b43c36914c7b7c198911ea8fb370a6124a784a6585d3e46104 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-bbbb9415 | sha256-f8ee2a9f7b1d8afb02925d5b39f5dc7d95c7c142f187d414ba46e68509018bfa |
