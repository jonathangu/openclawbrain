# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02a1da575e3574b7abbdd906c3a2e763180dabd7aec44faf80aa583f38ef8508`
- fixture hash: `sha256-5054e6a7c0e886819d5cfe411a8be2314f2663654e1f3235054d1d832296503a`
- score hash: `sha256-a55f0981e1bb0c2de2e47a020b3777b73784df66f63b85a37d5bd3e69d320f10`
- bundle hash: `sha256-cd363909857d0bba26187bc54220d606ee0128a494a28f32999ed22c6f5f0c7a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0f3e8434e68958d057893f328bab1455284fa045d15742e91d115a9a3a34202 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b6c4652183271047936d279c45af4c08287bc209f5947add85c7a13b16d225f1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-798a7d2c9c230d08da9e7e9bf15d7097be77c942d35e425d2415ccb393225b15 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-79d8c37bcc5eb84d31d69e8a301e0672b20c52b25dd99b602a5b51bf90ac50d3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab276ce8 | sha256-1293dbc3d785bfc649c8910fc3d35f1e3f2aa87f56b811d71e9b776d38834030 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab276ce8 | sha256-1293dbc3d785bfc649c8910fc3d35f1e3f2aa87f56b811d71e9b776d38834030 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0e36e59f | sha256-7f8b0a138174e3a4efea91443ecef92973f52dd5cd2d2e4dcd02f8d75942fe58 |
