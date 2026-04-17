# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-00bf2bd686f7cfc027d3b7749683ef5ae4ebe1a8b4b5f12763771285b87ec8ab`
- fixture hash: `sha256-1287af06cb4b83146712d66b78f07ce6e6ac74450d156f3cf86e05b95cfe0f1f`
- score hash: `sha256-f7347a1b3a96d649ae5132c94903bbc2eba63eecbb24e7342819d592dd406a7d`
- bundle hash: `sha256-e28ee8ade44695842418dd01782f42850951a1b6586de47c54730c662562b849`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80a7f9806b34ab4aca7f2c918d805e0ef978c8cb5147a44aad086817dfd7315e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-37eb71348698f6606660acab17924c932e373ff5c28a4ea0096bbccdb212a8e0 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-24efcb6c41779d8a119e2e046f0d08853611918e7338d0d3ec53dfcc7bb8b26c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-31db0098724851b3bc9d1724b92b39d3c6448216421c77fbff57aa9a79502507 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce7b5bdb | sha256-fbd6467a301cdc91520933b498f6ddf935e07221ddfeec6ec917a68f4448c14e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce7b5bdb | sha256-df607808d339ff7b6a50e3a16869f01b9070b924645a7f2875e30cc73a4afe98 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-adeb7524 | sha256-a760fa3f97e97201086fb5c6aea93cc655dec98636606c3d22b1aa6e09e75028 |
