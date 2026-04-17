# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adbfb582784ce9c57067bcd682b42040f9ff5a4fc2a41a6b215fa1e5e63926e2`
- fixture hash: `sha256-1b81b9ebc5b6e57a68ac36d63b63963fa7e0e03c9b05269658a97fc89e8025b0`
- score hash: `sha256-22f7cadaf4b71439f9cc4dd41963ffac406d98ac9633faf07c7fb3ea950d617e`
- bundle hash: `sha256-4965db5d770133b758c3f6a2a084b0200151f5e81c38f5077591cb570d1b7d97`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d918c89a43fc84e7a627af305c1d796a487842c9f1cf040b6474472ff6068ba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-52606b48b14cb39a7526cce867ce465fbadeca885f00cf10b05b5ebc914580d3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31de253f074a2eed94b7a15ae62a6048b54bd951ab7d749811079958f7aab58a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-55e90c3378ec94231f868ebb3af5a93508a812fce4d131fb3b1d0774d545aa8a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f584c601 | sha256-bac2fe9c7d51ba1d50163e29dbf8041601036494315d3d9374487fcec6f2a7f9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f584c601 | sha256-f4598250a352b2161aaa05667890dbaad8aa00dd4f11612196eefe2493109127 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2c558148 | sha256-948d5bfb8dfd564d9ef45a5ce69025de38425030a3edc0481045870835a34a45 |
