# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11c3fc8419968b3646c9393b68a3489cc8f59dd899a6b693d7a0be2f87ecb9f4`
- fixture hash: `sha256-de224889eeb5e399123549dc0a76f80a745378d0b306a7fbf4d142a78dbb77d3`
- score hash: `sha256-e719c328f852dc240204df84474b57d1a224fcb7b927ba79be6acae0382f96fb`
- bundle hash: `sha256-fbaed24c134292b3b8cd471d73c32991ee0b03fc8ec8f01dbdc927195e1583bc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6ec78224bfabb767d014be29075a2dce7e842b42fd97bfcab45b7a968e220fef |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-b5a5227c5566f20b724f1dc4d5807817eb1b9fbebadaa5a9031ceb03111aed83 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-cd1c2f3a2f3c80a35f609efe8a29691877066c9d5611c1b58e7115717ad8737c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-13f0cc28741cc8dc47fe4c063b564130119a12e89db8026cd2f65410dd2ffbb5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-627280e2 | sha256-68b41aadf031643baa539efe9ab3f886bf4d37c6af3b41f1d8557beed7521509 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-627280e2 | sha256-d65da4818285f0ca87cfc3b89491e7dbce1f2b96ac47ce4dc61b3b8905f58490 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-627280e2 | sha256-3725f717e6b24c190868a4d24f19dcd1662abd492118525278d3065e71eac5af |
