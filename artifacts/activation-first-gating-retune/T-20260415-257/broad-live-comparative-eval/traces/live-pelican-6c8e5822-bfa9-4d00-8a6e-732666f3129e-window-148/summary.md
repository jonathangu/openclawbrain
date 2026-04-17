# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-148`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304b5ee53cc148670256892da800bf0d31f07b699447be9e8eaaeff5a3c2cab5`
- fixture hash: `sha256-60dc2f86ac1ee754f931ba95c5a33382b613c3b1b0a7e2c96deb303d2eccd093`
- score hash: `sha256-68a7456dae5fe4f4cad06208552fc22a775041d2b0b696a32f012b678695dbe8`
- bundle hash: `sha256-df53294dc1311715e1b422dd9994a46a068ba02f03b56e77fb1966d957d7c590`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-00c0e12383f91dfc69316c04637d00981899188577451ad6f4512fc45d0cb8e4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8ff55310e14bd5acb4580702a06b9876a34c50121b4933e5ce6d7368444bd960 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d075a8e0eacaea0e292465fc55b4e5f198cbca3abf075d55133bf9dab41f031d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b62f559 | sha256-fedc3ca49938445f7c2ed65208f61449089062962add1608923afb72991363ab |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b62f559 | sha256-37b4682ff6bc9963366cc7a267c142f0f32c3d39b9a29af0477f0d044b86bda2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d531ce12 | sha256-e6552521b0d100ff05332113efa096858d25fcfbb861874da9940d34e51064ce |
