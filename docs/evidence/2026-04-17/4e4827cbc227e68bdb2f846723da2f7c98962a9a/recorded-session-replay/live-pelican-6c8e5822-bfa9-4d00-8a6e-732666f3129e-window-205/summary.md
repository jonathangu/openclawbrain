# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-205`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f579c26265d9087760a95275a1ed5d3c29a7fa2a5745f0cd5985ac21a42da923`
- fixture hash: `sha256-344c0f8fa42bcaf494090e8fb4c4629c475783bea29bc527602ae9b6d23e9791`
- score hash: `sha256-04877b061cf437557ce1d474ca9e38a6bcb81187ecf43ed5b18b6871c342c554`
- bundle hash: `sha256-6b6dc3eeee47d14223d771ae9c1a810052316a5213bcc5f6fa2992b115d3f9af`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c8dbb3069efa791ffae92b155399c68bb15a7550a719f9b3772c99bdfa5fdc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9082a47ec08c86203b429aa19739406fae2e0ad42b0095549440f80bf05a19da |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-08651df3bcf292aa7c6ffeaf273f2db72c9f53e7be2e34aeda2041473dfd3f2d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-278cfbdde29e0dbdf2edb497b9a85352bc44c585f8f6b961bcfb5a41c2415607 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e29c1262 | sha256-c2ae614e47536750077a1e8d208a878446af8b42ff82788f5eb0de9a0e1d07e7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e29c1262 | sha256-8a2965a52e35accc32b7ebc83d8b2aea0edb8edbe20e9c14739e84024641e87f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-25a16eb1 | sha256-07212c5b7bb82ba7af800ae1efc0d8f06bd3c3af38708dbf6ca943d581333889 |
