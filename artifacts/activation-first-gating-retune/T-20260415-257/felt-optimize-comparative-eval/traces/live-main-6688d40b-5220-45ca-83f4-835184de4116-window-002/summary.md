# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2ed84d12aa80219c71f67ac4b4dc49c9c31220d644ee2203f557cfbb2718f653`
- fixture hash: `sha256-a5d98f20c022a45dcdc79196fa677af12fe3ae7a1d81ee01512e8a79553eb0a0`
- score hash: `sha256-74514ba9b1dbf87694fc1e341836b4d0e28a38ff82704e7b0a42273845a0ee99`
- bundle hash: `sha256-19e7e4e703e27e4c07878f63432c1b1395d66855e58a07c42c5db7dce863a997`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d0401b09442b39c74248dfb10f1b77d9b52939def1349d2c685ecac4f520b39 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31210116fe667ef9c70491c061fbcc1fe1a2cded61f2550f7d26dd21175e4099 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-14cc46fc5b75d4d42059623cf1c8fcb5eb21dc83356b3de16ae82580fa5d5723 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c6e0c4ae6c02143e35c2432558db5223b1be02d5ee4bcae34c2b4b100e7cddc5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e7085b97 | sha256-5ba03bc5cafcdec4e17ecba4c9e8c000a9b6a3fc6d3f36ccf1698997021a4f8b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e7085b97 | sha256-1b777a0e9819b9ffb2daa1c88728978023bde8adb63127014b5effbe418cf265 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fb7ccffe | sha256-75bcf61b789bbacd2248bfb7948329b52362bfa41191021e721ae48e8afdb2ce |
