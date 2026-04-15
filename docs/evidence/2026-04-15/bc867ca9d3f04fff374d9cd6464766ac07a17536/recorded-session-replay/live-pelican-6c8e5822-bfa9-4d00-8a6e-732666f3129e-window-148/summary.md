# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304b5ee53cc148670256892da800bf0d31f07b699447be9e8eaaeff5a3c2cab5`
- fixture hash: `sha256-60dc2f86ac1ee754f931ba95c5a33382b613c3b1b0a7e2c96deb303d2eccd093`
- score hash: `sha256-23da17d9ced75eeb4e6053ba95e9612df67dec924ed8596572a19c7b7840454d`
- bundle hash: `sha256-370b71a2a1123e853fe293e253d6ea1abed8aa723aaa3c5ca1ad472ca9278d4f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81d1a7582801981771b9bc27a32c83725b8a8a67e2715cd65f17099531df2d18 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7aeb8f783dadd7e68c8dd32ac2052fad4a8544870cac8ef6fce5b9d997596a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-69e01b31c68f5c2e58fdf78e701f161c9cf42fa94edbfa44edc6c2e776327146 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2ed973f9ab31bf0ec22148cf15d368940152e99263048bbd7c796d7a10e890b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-12c06f7a | sha256-6965b4062b7e40f61b53885aafb418069bf37a79aae3b0fb9217c1d760713cd1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-12c06f7a | sha256-536d00772bbf05c350abaa80e72e025146a28de620b787f55ca3f49673887544 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-12c06f7a | sha256-6965b4062b7e40f61b53885aafb418069bf37a79aae3b0fb9217c1d760713cd1 |
