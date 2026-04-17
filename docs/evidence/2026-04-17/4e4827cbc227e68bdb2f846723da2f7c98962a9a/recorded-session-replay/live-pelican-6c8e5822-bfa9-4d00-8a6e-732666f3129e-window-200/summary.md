# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200`
- winner mode: `graph_prior_only`
- trace hash: `sha256-438b689c90e1516f117c130a44f955ebe5121f19131ef3c8af4f3b72e782a392`
- fixture hash: `sha256-fef64d4e61173927de1b8c7e42759f7ee5918ab3e67738573626a046f39d5b5e`
- score hash: `sha256-1f2cb12713932e3a9137590447539351f4b5c28446df0140e6f6e2e42956d947`
- bundle hash: `sha256-f0f512badc5b39f691f5ef696431792cfda219e7d244ac6c598156442451de14`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3e6c6ba4308832d244620436e1eb71e4969051bd02e8a257e4c9a12dea8653e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-56fe2350860500ef8da1db7d571941beb41722f0547d69df34d1873f01b4a12a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d10ea8ade34bd3f6f273203bd82ad50178b7284902d363a60889a81d7d304c27 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-552516341c9ab6a22ba79ca0e85cd2efae1831b94ecc42cc1d0a28c6d4506060 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00339a6a | sha256-583459a5f644ec096c04101716dc9397c8cbda6eb1b5791afa30004011a199e3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-00339a6a | sha256-cda6803173a470edf8533369b9e58a08cdd4feb0ad035227a8863c21bf2995f8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6a5120b1 | sha256-f0fad61159b0d27dd2c5ac13a6df09e860dcf54516b9bd5ca96803ec436f5416 |
