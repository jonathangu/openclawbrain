# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bb64d18c7380c29a36adda7f18b9d94028bd2ec79c3f043249c311ff96079b77`
- fixture hash: `sha256-4d9c945d16c80ffc64625c9921a10c1aa73d0e2d0d7dc96750c287fa87ef0a3c`
- score hash: `sha256-eed3b86ca2baff703997ce36a90598f8236dbab960098b778bf7910629c898d8`
- bundle hash: `sha256-b9f204f38e2017a2dcff4b69b27ab14a777add57602354cf011d65da413b54c7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2057eeb74350d5472d7a207a6cd23d83fdfc1cbff7a9da70502d2c9709cf85fb |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cd0070a7a293a4bfa55cda151e197e56a08da753035498083ffe1fb76e9fdbaa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5768b43263944f3ce8e4bcf64431f33719ba7982d4124df40b5d7e05963a300d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ae2357a5d20b43b4038aa6a2d2c634cacc51a9dd3465e7839e8bd52c7f241e2c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef671310 | sha256-e6aea613b77f8e2304e975204f9f8f5820371d04a6ba2bbd15834a8ffb8665d4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef671310 | sha256-ccfe1c69eb0961e21c08c8398f4193bc5270f2a58d7831b981015835e8c5dc0c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b535cecf | sha256-05a10a14c7bdb08bf3497590500af807e27d0b6cc230d28253dc186057d0765d |
