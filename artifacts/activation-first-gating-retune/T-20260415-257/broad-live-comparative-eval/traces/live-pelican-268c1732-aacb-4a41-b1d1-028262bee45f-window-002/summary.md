# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6225db554a4364f98319fe1020858f0210ac404609a5303b38b0c9b7d31d658`
- fixture hash: `sha256-ed619df46c192591981773c10de0d36b74c9e5dca7f78e3181e7b1aa8b066c66`
- score hash: `sha256-5a028cf4fb0d9379a48936373486cfa238253cab7de99fdbb48d440d329222d2`
- bundle hash: `sha256-403002c4788afffcce3756b26bd36ccb5143546c2e756781c08d44209055d545`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88c859d8e45ae0f9acc890628cb16be0104a115fdd6d7748c85da1fbfab29bae |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0e4149d59f6c72d2d65cdabb1c8aa7d2752c3b50336c05d560c82ddf952135a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b305dc3009361caba5cae487b7d9846449b15b3c790033cbbaa39ca4ef856625 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-20ebfd20f7892bbddf4a84f69e4ff5977fc6dc1bc40f0219400b8eca07882cb1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8c0bc3d0 | sha256-64ab0adad42bebd9ec264e7db2bed9f12e53bef6adb34229890e05043b21958d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8c0bc3d0 | sha256-64ab0adad42bebd9ec264e7db2bed9f12e53bef6adb34229890e05043b21958d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8c56a5b7 | sha256-990795d459b65dee5eed600abcd5fe1a5030438d3f8b9ac3d64f13ff2d0a4596 |
