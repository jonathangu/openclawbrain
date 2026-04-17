# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-14ade459ea986baa6e4e71bbbde0e89dc1fae7980400ac765d36815dff4c4f35`
- fixture hash: `sha256-9c30c978d165bf9a25e14aa9b77d9a12a45f7a9014b4a8204bd05ec1ae139d4a`
- score hash: `sha256-ab2d75d7defcbb6b1ce2182832cbd77e9027f21d94901a4882de2111f9c67f91`
- bundle hash: `sha256-2d326462b1e42a5225d136672f560f35202c1893e7fea2516d79d1a504d3a360`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-334c135bfa30ec156738872f694abf9297995f829f0e8e1c5041f315be0a98b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9591e772ca7e1fc0a719b5d0656ca408a7599c14f06f938c192c88c79ad761e4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-40830912f61def3c149332648a3d0f04d0136513034c3bfb6ee5e7791a024420 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-17a19d5e8754c51c2d1b10c7152542c93766a7bcfddf967ec602b04847dbfbf5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-55f614f7 | sha256-fdfdafe0e905a78fec623c867c90722b813322d7d7c08d574f0ebfe1e0a0276a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-55f614f7 | sha256-e315bcd4b3474e2c9fd672972a817f25ffb657dc167e4160cbb97cce08d15ef8 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-95cc017a | sha256-1814f112ddd619d626c984c38c4179add8685aedf01b99abdeb6b326fc777825 |
