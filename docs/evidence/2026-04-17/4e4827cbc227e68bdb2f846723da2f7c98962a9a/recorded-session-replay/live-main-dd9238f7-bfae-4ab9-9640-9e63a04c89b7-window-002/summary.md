# Recorded Session Replay Proof Bundle

- trace id: `live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f45e438e0f99f9d56b4e0ce3ef341383c4f2368651efec5583a2c7447c8a5e0`
- fixture hash: `sha256-24e221e1cec238f614a332fafbde124000574c7f4eca983f394d512d73646f16`
- score hash: `sha256-5cf9994e827dfa0edc84ecc42694fe6b3c6fb11e2ddde1f4b17ff89d9c960ab0`
- bundle hash: `sha256-ccce48e013f6f01f7e12eac45667acf197f082c4e5b24cf85056a8ac1ff596bf`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5624a20b92c6a5c4c5d269dbfed46d621fb3009b7407cdb61d3d2abad216a892 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe59fc6923a45112a2f2b6731c336406c26c0619cacd577d1797524c5178f470 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7216605802caef65f64eba38befcc1513a92b0cce1a4a7a432c132e49bef4e40 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6c4ceb544de226e4a6e8880bd982e355b3251392cf39a6239e725639af0f08ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-087ed0ca | sha256-31dd431aa8b8497f4570e65e857051c1f0f1b80181a12fb8af2c689aa17c093e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-087ed0ca | sha256-cbce97446e96243523bd8a3552887b09286c21ab43825f55cee720e55b1880c6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f6085719 | sha256-eb8bd1d8456ba45084072fc200775a669f441c5fd758876fac89b5387f90818c |
