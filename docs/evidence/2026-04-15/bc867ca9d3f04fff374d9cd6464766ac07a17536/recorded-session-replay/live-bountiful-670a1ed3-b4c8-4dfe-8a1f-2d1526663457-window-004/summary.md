# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf10cae0f36c648f32a6c50dd8217f4092591b4c33fa07516441723d723d101`
- fixture hash: `sha256-74ccb99a45cbecfbd0675ba926480f518b6d9257f4cbecb8a7eccfb5e3bc826f`
- score hash: `sha256-5b1a213400175fc7ec4c6f61f4c3a91bfaeae098cbe01a3399c73c9c6bf08fe6`
- bundle hash: `sha256-b294d512e224f21da90206cfbe34a564f56a3ac25a9463e2d13c4d5ef6ecb90a`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-deab881832958a8bd935ee6c81daddd68f45a0ca219749d213e1a30ab0bb8c14 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c048dd00138266cb810af91c3dc6cbfb4d0f78954ba9177f5f9c3ec7e737c786 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8c1f99a63ed769c8b1fae4ef494cac6fbc4dd6010384b017748e418685961b1c |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-296965f4b23a0ac086ab32b1f643ab000b69f40177bbc94472291124d9a9ea47 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-244fef5a | sha256-ebe8ea05525b592dea52a6bce7b34fd6c6a09e4ca90fb9d126e49a24db111640 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-244fef5a | sha256-f9e616023171b831e423dddf8a7308c4926114388d844f434e5535f838ef5ee7 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-244fef5a | sha256-ebe8ea05525b592dea52a6bce7b34fd6c6a09e4ca90fb9d126e49a24db111640 |
