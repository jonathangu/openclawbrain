# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304f24fec2ef73f307ee3b3bfeff3d4bd90894e7d9fa693794ad2f916befa2ce`
- fixture hash: `sha256-0d0f5f0a3dfc50799aa0a0583bb1e17204f3f01f50323b91030ad8276022d234`
- score hash: `sha256-e277687fd8c8ca969dfd924d10cdd1a6997f51575327366bcdfc6910bedb6802`
- bundle hash: `sha256-9ca7613e9be3a91e93db63c44e605d7b16cdf4ccae0673ac25bdfa1e77ad8113`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2db02be6aed4a54b1a82eb4486629fc6a8c812b69fc8bb1feef7e61858ba9b3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-93705ee1945d184811d688ea43f90700977eb0aa6ba96e25e7c282ada1037566 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81cb2b53b8ff5b75eecca02c0d8c7828b54338dbfea329fd0941abab5095e492 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-09eed067f732617400e678b9e705d0ca5d673c63e89cac6c2eebd6a5b4adb5fe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-da8417cd | sha256-047de484cabc29405934d4309ae9f8cc1020b4d369ad6a1a9303f3b1e7c6badd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-da8417cd | sha256-27b51d632c05856a6399f6a17232271bad0d32b20c1da80968ea25a313c66213 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d6561b34 | sha256-636d484a0c61fe0ebaeea52198ab5e161af23da1526406fae2ecdcf7f66c5999 |
