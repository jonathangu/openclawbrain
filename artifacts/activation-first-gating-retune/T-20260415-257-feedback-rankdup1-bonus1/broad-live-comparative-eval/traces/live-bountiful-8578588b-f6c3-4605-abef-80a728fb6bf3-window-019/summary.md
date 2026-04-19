# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6585559519673a3db32bffae40afa6ee2742e112449b33d0923762a2de179a50`
- fixture hash: `sha256-e5d86f08bdcfbae469f4662e91d9d271a78451f428781b94a6703c49ec68efae`
- score hash: `sha256-fa377f04f9842de101d56edde3ced2613f61b07f0a3bc8e49703901c5e009cf0`
- bundle hash: `sha256-34be7944d7baf29fbc9fc7bde1b8d34b366d3627c5eeb9915453089cf4e5e035`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ad1968bf09244474e138b2781f33e5606f4a1c015708e46d2c74447f4594a893 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f428a71793b12ef5a9da5dc7acad2d82e941e4d9303ebf0e1fd54ebe22266b1e |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f07f22c5438706a63e6c6ebf4231aaacd4bb95c875f47a265844d8d022db2695 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8b8a540ea4c67a4f87104570c2021b305c3c4553b0aab078040a9cb67964b683 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-74671652 | sha256-ccfd8990e99fadb9c5ecd22bc523cd45badaf43130b80cecdb59eab2b84f25ad |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-74671652 | sha256-5018648e06a9467e4c2ef1233dcafa2b542c6e5f9258bd29b408bca541f00608 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-74671652 | sha256-45da2281f436206ee35f847c38bd8d4a13af176cc8e98736b03ddb339d1c2792 |
