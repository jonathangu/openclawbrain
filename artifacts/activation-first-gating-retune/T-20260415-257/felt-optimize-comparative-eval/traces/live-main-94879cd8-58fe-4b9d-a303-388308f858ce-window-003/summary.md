# Recorded Session Replay Proof Bundle

- trace id: `live-main-94879cd8-58fe-4b9d-a303-388308f858ce-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ab9e7c92fecacf60147f27b8c27fc049edab247767d390d6cfd938c3433d0a10`
- fixture hash: `sha256-924c4ed1186166bf0f6b342b0967c241c06f6d79d3f88b2eb96f947a6b1061b7`
- score hash: `sha256-da4af39e6621cbb3f7b68bea5515076d5f581d7fc5013b8b2c47dbb9af99aeb1`
- bundle hash: `sha256-228f79a854b0fab30bfa3cf9dab892576006ea7d330a04de57d7eab43f45fffb`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af4d866d80b0149b53a0921726b1499466ad574098653e689171c2b2c56dcca1 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-821d2011debbb86081c01d94f6edcc2817cdbcded9577f2b856cc9a180afbbab |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-17289a99ed66a337038d09921171ddf136f9ceffef12ca4811f844671213074f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6e6798bf51cee65a2cf6e16368fe40423fcd10b9e5ad2c0f4fbc4d438be31ccc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-eea2332e | sha256-78b0b248752e0cddc710877918f9be6939b542c3bea99da18425893287e4a637 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-eea2332e | sha256-78b0b248752e0cddc710877918f9be6939b542c3bea99da18425893287e4a637 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-eea2332e | sha256-844dfde2e51140140830b6f62d460d98189245005fe193b3806b2b8272d46015 |
