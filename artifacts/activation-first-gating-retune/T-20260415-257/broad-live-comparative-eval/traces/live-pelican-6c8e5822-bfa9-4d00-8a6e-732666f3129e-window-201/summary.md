# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0daac77494acc6aca056ecf3e9f12fd58f33e4988f5a646f0ba5dd6ef080149`
- fixture hash: `sha256-267dbae4de2075656207ffdf48cdf822d6c7cd1996c42f8535ce786c53f3660f`
- score hash: `sha256-280c6abb4a6523aa2ffbd318315a8f4641afd82e628f753d45c68a4751f1eb94`
- bundle hash: `sha256-0748d034d857446ee0e595b0161ad00ef8c35b0619625a5e69620b1e26d18a4d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fed91035473e3fc2f9947814563b07b47451ce7ca5cc7e497e3f1c68f58c389 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e4a3d36e96c53e5a4e39324452b629b40304735c4b84bbca275c4385d54db074 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-391f906d9ccde76cc323bf8d38fe283c471e56a455be8daede3002e6c49e0aee |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4e7de0f9fed8deb1057e87a26d81e73b2b85aefda938f4f25e6e7215d0566465 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60b38fd8 | sha256-a6d281edc051bf908a4b0881cc753560b69047c168e636fc969462f965bb538f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60b38fd8 | sha256-2ae984c26a8dd89f792f88bc211ef3a9c296efcb754deff25fcf1d98902a7f43 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e3903a89 | sha256-df047e848816df0881f0f7140063ccdabdb1b9d97d9537238bb97bd62365b0d6 |
