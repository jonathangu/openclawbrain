# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304f24fec2ef73f307ee3b3bfeff3d4bd90894e7d9fa693794ad2f916befa2ce`
- fixture hash: `sha256-0d0f5f0a3dfc50799aa0a0583bb1e17204f3f01f50323b91030ad8276022d234`
- score hash: `sha256-785485e5a22fbf04ed66256f47498203f2572f97d7aff6a400a6af00955531cc`
- bundle hash: `sha256-9bbde6e43ac2729a1271b2f7abbb2ca7cd7a81b472b3fb0907e504d2612c4743`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2db02be6aed4a54b1a82eb4486629fc6a8c812b69fc8bb1feef7e61858ba9b3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b322d5674a96a65e517aa3587b6f116325d56842ccd8664866d8b106fc7daa6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c22fc019428242c7f13f331bd23ce61ff519deed4b495f37472d6335ee166f93 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0a94fe7c32bf5436998e3f433bf909d900c6473af0aef1391659f44777156607 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a86f5d22 | sha256-a50af72999ca0f21ac953f140edafb47967164b693211f5758e81d5235a7d731 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a86f5d22 | sha256-70e6a6e813f825e8399b937b48078f1abe0a7110f5fdc524983ca474ad7933ea |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a4416089 | sha256-e03b371b6d25aa166aa36d0623938619377f41f2139cbcde803e87118932463d |
