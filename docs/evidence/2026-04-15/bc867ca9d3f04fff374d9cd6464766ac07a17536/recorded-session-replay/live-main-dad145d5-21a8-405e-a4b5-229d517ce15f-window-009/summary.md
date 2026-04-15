# Recorded Session Replay Proof Bundle

- trace id: `live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0f24d3812e038d9d2b67d9309de9db96cd24c2faefb0b5dd93caf569b3c1d1f`
- fixture hash: `sha256-6b6c634b067ee2b84c6981ae8fc0d6c41efb6194e1723d88dd7d0087036cd1ac`
- score hash: `sha256-4f5a8ff64fba82510bea87cce10059b680d42f22e12b169d3555aa9bb2272a33`
- bundle hash: `sha256-874e889da065eed131324627ced11bf9c4f423429789793f823b56d2309b2350`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ea26975a8365e08832501930a2890706222216fe363c833adbd0065a774a3f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e814979856f144e66135444c887d2c9bb271a0172d56497ac131854ed9952972 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6ce733ff02fc96ca31899c36ac73dcc40c5ce00076fdad28eca1db041813d566 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-755a463f4ef84c4d49a36207af2d7dea25266b47587310811147812e1a907250 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-149fe5f4 | sha256-ebdad761ad0b0509871fe2326c8bc11722e61bc8d86b59a823691795dec3d4ff |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-149fe5f4 | sha256-41856ba648ad422e2fe378000bac28d1eab07cecbbf60f3ee68133e84d62792f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-149fe5f4 | sha256-ebdad761ad0b0509871fe2326c8bc11722e61bc8d86b59a823691795dec3d4ff |
