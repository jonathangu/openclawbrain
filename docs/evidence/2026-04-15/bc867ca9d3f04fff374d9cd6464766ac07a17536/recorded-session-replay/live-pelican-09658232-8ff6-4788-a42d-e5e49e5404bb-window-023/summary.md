# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f60100eb1742bfc6c299af2f5afe9b6c211473ff986af1ecb211b198ec2ac6e1`
- fixture hash: `sha256-7060077aa89ea2d2ed121c14a4166c1764801c149a1d2df1467761d22c2169ae`
- score hash: `sha256-64f35244bcf00fc645777e38437e3b7dea2dfadfc680dbbcc400f1e5159a9f56`
- bundle hash: `sha256-4a189a72176a7ad7f5dc0b96d99cc41578c7cca274cef3c823a9d564cba51751`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1af67f3f0f2a5d2c63ece4b570453604d2bc85441d7219830f849b19b9d0d604 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa0e8c6ab9ce3f463bbb95eba8d3ae984fff9cca1f92959c98bb25ecacd8b8bb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5b832bb52095ea995d0598e552fde41a50aff38c706ee63e994a6ab579947b24 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c3d87c02c5486184ff1b4032bc0758675925a4bebef53f97ead4c59a130418a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b96507ba | sha256-b82e685f993706b555277ffb4fad84fae14af14969ddb80746cd1a2514a58fb6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b96507ba | sha256-fa48214c991ea96bb3405222898e5aa0084a5752c2b6c3b00d34a7f636ff626d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b96507ba | sha256-b82e685f993706b555277ffb4fad84fae14af14969ddb80746cd1a2514a58fb6 |
