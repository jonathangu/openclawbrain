# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb16877a0fe2caf32819847f224e56592583ef2fd1a04c845d04e2ee17b64d0a`
- fixture hash: `sha256-4ba8498860fc7c42d2e5ff1842f641b3036471f0db760ed3478d90908a631234`
- score hash: `sha256-bf56aac9a86157239c3b9ddffaa6eeb6471a838be28178a898c91622f573db4d`
- bundle hash: `sha256-c34882df2472a9df4152a99fb2fffe8cdb61c56b624412ce8b52b3076056406e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-09b23957d62d4dd2aef54d6f6e2af1d61d598bff57f4a467b714b73990a75fcf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-43bdd20afd10975041a90a68d84a9920cd049934cdfa0b0df845eb1bf6dcc19b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68108b0d2b35b5a989e391fb0cab714adf97d47dbd3f5a7c341a63afd52f9355 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b8f9b82a64aaba92ee0fe642e7f5ab4d439362f99dc157eec2ef64ef2c880479 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1f21753b | sha256-133d1f1ed12140fe7892769815764b9f85e4ab446ef77aa2a937ae5e5659669f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1f21753b | sha256-64661fc9860e81a761dff808f6e96e86d4f8c454c6efd62122a86527871f82a9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-873fbc10 | sha256-e4f06711c6c6db0951c62843bcc750a5c9498edfe97798e3ea3e72f863fc6a5a |
