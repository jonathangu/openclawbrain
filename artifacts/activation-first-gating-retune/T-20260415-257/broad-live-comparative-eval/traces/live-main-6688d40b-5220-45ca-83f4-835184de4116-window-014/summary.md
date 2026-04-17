# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-299442dafe9170b6e7846efcbc8b3fa1c668db88067ca331f05cbc368b7140b3`
- bundle hash: `sha256-6a15c9d5c9f6eea591cfe085825409637d1eac5367360770efbb15f10ad81307`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-20f50504408046e7b728bc088eb1edb0d96a1aa90e20b8b7b6428e37361bc938 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d05fb054bfa2ac00e071ec63448143385c6dac5a01f6b643909177b0a40dff22 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a181bcd820ff80f0d7e4a642ae7d23328b0a756d6ea5878e47b33149167e429a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-992df713 | sha256-e672a18bda64ef61d9f529873f2564db0fc434be63e82489b18f852165637fef |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-992df713 | sha256-1d3a246b806c076788a70d800f9aff53fb210bae1c64ca5c9b1d70c56f8194e5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ec58b312 | sha256-99282eda55063c3f5bcc67a35331727eee4b8f4a94fdcac20e1ce0c4fcfddda4 |
