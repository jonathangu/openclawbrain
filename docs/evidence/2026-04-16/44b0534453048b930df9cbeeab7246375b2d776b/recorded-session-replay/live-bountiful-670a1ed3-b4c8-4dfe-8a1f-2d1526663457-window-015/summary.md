# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06fea57a3516f2a337e636c80dde5aa0f7b5c4e7b115ef7c15ef4879727a06c9`
- fixture hash: `sha256-0bd1e90ada8a113768901038367ef3359fd513f44e7b3d01e72effd5c2301b57`
- score hash: `sha256-ce8ef2228eeb1ffe286b5191d94082cbf7e79305de83d64d33ce4b82c79d976a`
- bundle hash: `sha256-1adb96d59f0d10a0063aa7359ab23231c90946292bab7f3de486f83019c11b30`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c5ee087496e1c83dd50cdb77e530bafbdd0a3348e86d19deb3da1e266821f9a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b0f74d292e7cb2d8d5a5396515ccab3100c7696b358c69692aade551f5cb566 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1792bf95d35df5c0cb03ab8231f1b8a99481f022deea8d91617618e481601242 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-548751171008bb856e198edf6f1bb894947380eca4ad4c1daafc2c735ddaaa96 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c2d879 | sha256-6f531d9e00779b6563c76a916e8186567b491987669f59a4b59a5564ad914f9c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-83c2d879 | sha256-4eab073da554770c1d6a410fb145bf0831e294a6208c1424f94970c9407c2ecf |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-1c28d282 | sha256-e14375f997dbaae0c74165cb9c2359fa484c3406dfa311b3be78d847e5261198 |
